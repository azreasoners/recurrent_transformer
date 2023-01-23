"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from ste import reg_cardinality, reg_att_sudoku_c1

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.C = self.f = self.create_v = self.tok_emb = None
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.causal_mask = config.causal_mask if hasattr(config, 'causal_mask') else True

    def forward(self, x, layer_past=None):
        if isinstance(x, tuple):
            x = x[0]
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.causal_mask:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att_to_check = att.clone()
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att_to_check

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        # x = x + self.attn(self.ln1(x))
        att, att_to_check = self.attn(self.ln1(x))
        x = x + att
        x = x + self.mlp(self.ln2(x))
        return x, att_to_check

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        if config.tok_emb:
            self.tok_emb = config.tok_emb(config=config)
        else:
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_classes, bias=False)
        self.losses = config.losses
        self.all_layers = config.all_layers
        self.n_recur = config.n_recur
        self.hyper = config.hyper
        self.C = config.C
        self.f = config.f
        self.create_v = config.create_v

        self.block_size = config.block_size
        self.test = {
            'n_recur[cross,uec]': False,
            'n_layer[uec_last]': False
            }
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        logger.info("number of trainable parameters: %e", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, idx_ulb=None):
        """
        Returns:
            the loss as a scalar
            the logits in the final prediction; (batch_size, 81, 9)
            the attention for the 1st data in a batch; (n_layer * n_recur, num_heads, 81, 81)
        """
        b, t = idx.shape[0], idx.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # collect the attention matrices and predicted logits
        atts = []
        logits = []
        for _ in range(self.n_recur):
            for block in self.blocks:
                x, att_to_check = block(x) # (batch_size, 81, 128) (batch_size, num_heads, 81, 81)
                atts.append(att_to_check)
                if self.all_layers and targets is not None:
                    logits.append(self.head(self.ln_f(x)))
        if not self.all_layers or targets is None:
            logits.append(self.head(self.ln_f(x)))

        # compute losses
        loss = 0
        if targets is not None:
            # 1. compute losses on predictions
            for logit in logits: # (batch_size, 81, 9)
                loss += F.cross_entropy(logit.reshape(-1, logit.size(-1)), targets.view(-1))
                # compute the constraint losses
                if 'c1' in self.losses:
                    probs = torch.nn.functional.softmax(logit, dim=-1) # (batch_size, 81, 9)
                    probs = probs.view(-1,9,9,9) # (batch_size, 9, 9, 9)
                    L_c1 = reg_cardinality(probs.permute(0,3,2,1).reshape(-1,9), num=1) + \
                        reg_cardinality(probs.permute(0,3,1,2).reshape(-1,9), num=1) + \
                        reg_cardinality(probs.reshape((-1,3,3,3,3,9)).permute(0,5,1,3,2,4).reshape(-1,9), num=1)
                    loss += L_c1 * self.hyper[0]

            # 2. compute losses on attentions
            for att in atts: # (batch_size, num_heads, 81, 81) for Sudoku
                if 'att_c1' in self.losses:
                    att_p = F.softmax(att, dim=-1).reshape(-1, 81, 81) # (batch_size * num_heads, 81, 81)
                    loss += reg_att_sudoku_c1(att_p) * self.hyper[1]

        atts = torch.stack(atts) # (n_layer * n_recur, batch_size, num_heads, 81, 81)
        atts = F.softmax(atts, dim=-1)

        # compute loss for unlabeled data
        if idx_ulb is not None:
            # forward the GPT model
            token_embeddings = self.tok_emb(idx_ulb) # each index maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # collect the attention matrices and predicted logits
            atts_ulb = []
            logits_ulb = []
            for _ in range(self.n_recur):
                for block in self.blocks:
                    x, att_to_check = block(x) # (batch_size, 81, 128) (batch_size, num_heads, 81, 81)
                    atts_ulb.append(att_to_check)
                    if self.all_layers:
                        logits_ulb.append(self.head(self.ln_f(x)))
            if not self.all_layers:
                logits_ulb.append(self.head(self.ln_f(x)))

            # 1. compute losses on predictions
            for logit in logits_ulb: # (batch_size, 81, 9)
                if 'c1' in self.losses:
                    probs = torch.nn.functional.softmax(logit, dim=-1) # (batch_size, 81, 9)
                    probs = probs.view(-1,9,9,9) # (batch_size, 9, 9, 9)
                    L_c1 = reg_cardinality(probs.permute(0,3,2,1).reshape(-1,9), num=1) + \
                        reg_cardinality(probs.permute(0,3,1,2).reshape(-1,9), num=1) + \
                        reg_cardinality(probs.reshape((-1,3,3,3,3,9)).permute(0,5,1,3,2,4).reshape(-1,9), num=1)
                    loss += L_c1 * self.hyper[0]

            # 2. compute losses on attentions
            for att in atts_ulb: # (batch_size, num_heads, 81, 81)
                if 'att_c1' in self.losses:
                    att_p = F.softmax(att, dim=-1).reshape(-1, 81, 81) # (batch_size * num_heads, 81, 81)
                    loss += reg_att_sudoku_c1(att_p) * self.hyper[1]
        
        return logits[-1], loss, atts[:,0,...].detach().cpu()
