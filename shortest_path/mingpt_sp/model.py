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
import matplotlib.pyplot as plt
import numpy as np

from mingpt_sp.sp_ste import gen_node2edges, reg_sp

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
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

    def __init__(self, config, layer_idx):
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
        self.register_buffer(
            "mask",
            torch.ones(config.block_size, config.block_size).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.plot_attentions = config.plot_attentions

        self.plot_attentions = config.plot_attentions
        self.layer_idx = layer_idx
        self.config = config

        if self.plot_attentions:
            self.subplot_width = np.ceil(np.sqrt(self.config.n_head))
            self.subplot_height = np.floor(np.sqrt(self.config.n_head))
            self.fig, self.axs = plt.subplots(
                int(self.subplot_height), int(self.subplot_width)
            )

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # =============================================================================
        #         ADJUST PLOTTING HERE
        # =============================================================================
        if self.plot_attentions:

            # self.axs[x_subplot, y_subplot]
            for head_num in range(self.n_head):
                x_subplot = int(head_num // self.subplot_width)
                y_subplot = int(head_num % self.subplot_height)

                self.plot_cell_attention(
                    x=1,
                    y=1,
                    att=att,
                    head=head_num,
                    x_subplot=x_subplot,
                    y_subplot=y_subplot,
                )  # example plotting the first head's attention for gridcell in position (1,1),
            plt.pause(0.01)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

    def plot_cell_attention(self, x, y, att, head, x_subplot, y_subplot):
        unrolled_idx = (x - 1) * 9 + (y - 1)
        self.axs[x_subplot, y_subplot].imshow(
            att[0, head, unrolled_idx].reshape(9, 9).cpu().detach()
        )
        self.axs[x_subplot, y_subplot].set_title(
            f"Attention Gridcell ({x},{y} is giving, Layer {self.layer_idx} Head {head})"
        )
        return


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

        self.plot_attentions = config.plot_attentions
        self.layer_idx = layer_idx

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        # input embedding stem
        self.tok_emb = nn.Embedding(4, int(config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(
            *[Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        for block_idx, block in enumerate(self.blocks):
            block.layer_num = block_idx
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )  # -1 for to remove 0 output for sudoku

        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.plot_attentions = config.plot_attentions
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        logger.info(
            "number of trainable parameters: %e",
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )
        self.config=config
        # self.node2edges = node2edges
        self.node2edges = gen_node2edges(config.grid_size)

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
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(self, idx, targets=None, steps=32):
        self.node2edges = self.node2edges.to(idx.device)
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        
        x = self.drop(token_embeddings + position_embeddings)
        intermediate_outputs = list()
        for it in range(steps):
            x = self.blocks(x)
            if 1 and targets is not None:
                intermediate_outputs.append(self.head(self.ln_f(x)))
        x = self.ln_f(x)
        logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = 0
        if targets is not None:
            loss = 0
            num_outputs = len(intermediate_outputs)
            for logit_idx, intermediate_output in enumerate(intermediate_outputs):
                constraint_scale = 0.5 ** (num_outputs - logit_idx)
                loss += F.cross_entropy(
                    intermediate_output.view(-1, intermediate_output.size(-1)),
                    targets.view(-1),
                )
                if 'path' in self.config.losses:
                    probs = torch.nn.functional.softmax(intermediate_output, dim=-1) # (batch_size, 81, 9)
                    loss += reg_sp(probs, idx, targets, self.node2edges) * constraint_scale
            loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
