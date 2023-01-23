import torch

import torch.nn as nn
import torch.nn.functional as F

import math
import logging

from tqdm import tqdm
import numpy as np

from torch.utils.data.dataloader import DataLoader


logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model.to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def load_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            logger.info("loading %s", self.config.ckpt_path)
            ck = torch.load(self.config.ckpt_path)
            ckpt_model.load_state_dict(ck)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y) in pbar:
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = (
                        loss.mean()
                    )  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()

                    # report progress
                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}."
                    )
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss
            else:
                return float(np.mean(losses))

        eval_list = []
        for epoch in range(config.max_epochs):

            train_loss = run_epoch("train")
            if self.test_dataset is not None:
                test_loss = run_epoch("test")
            if epoch % 1 == 0:  
                total_correct, total = self.give_exam(
                    self.test_dataset, batch_size=16, max_batches=999, steps=32
                )
            print(str(total_correct) +'/'+ str(total))
            print(total_correct / total)
        return eval_list

    def inference(self, x, steps):

        model = self.model

        logits, loss = model(x, steps=steps)
        single_pass_pred = logits.argmax(-1)

        return x - 1, single_pass_pred  

    def give_exam(self, dataset, batch_size=32, max_batches=-1, steps=16):
        loader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        gc_correct_nt = []  # no trick
        wb_correct_nt = []
        gc_total_nt = []
        wb_total_nt = 0
        given_symbols_correct = []
        given_symbols_total = []
        total_correct, total = 0, 0
        for b, (x, y) in enumerate(tqdm(loader)):
            x = x.to(self.device)
            y = y.to(self.device)

            outputs_nt = []

            pred_trick, pred_nt = self.inference(x, steps=steps)
            outputs_nt += [pred.unsqueeze(0) for pred in pred_nt]

            total_correct_batch, total_batch = self.calc_acc(
                outputs_nt,
                y,
                None,
                gc_correct_nt,
                gc_total_nt,
                wb_correct_nt,
                wb_total_nt,
                given_symbols_correct,
                given_symbols_total,
            )
            total_correct += total_correct_batch
            total += total_batch

            if max_batches >= 0 and b + 1 >= max_batches:
                break
        return total_correct, total

    def calc_acc(
        self,
        outputs,
        y,
        givens_numeric,
        gc_correct,
        gc_total,
        wb_correct,
        wb_total,
        given_symbols_correct,
        given_symbols_total,
    ):
        total_correct = 0
        total = 0
        for y_idx, output in enumerate(outputs):

            total_correct += (y[y_idx] == output).sum().item()
            total += len(output)
        return total_correct, total


# =============================================================================
# MODEL
# =============================================================================
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
        self.layer_idx = layer_idx
        self.config = config


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

        self.layer_idx = layer_idx

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DigitConv(nn.Module):
    """
    Convolutional neural network for MNIST digit recognition. From:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self,output_size):
        super(DigitConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.mapping = config.mapping
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # initial linear encoder

        self.digit_conv = DigitConv(config.n_embd)
        self.LE = nn.Linear(28 ** 2, 128, bias=False)
        # transformer
        self.blocks = nn.Sequential(
            *[Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        for block_idx, block in enumerate(self.blocks):
            block.layer_num = block_idx
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(
            config.n_embd, 20, bias=False
        )  

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        self.config = config

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
        b, t, h, w = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.digit_conv(
            idx.view(-1, 28, 28).unsqueeze(-3)
        )  
        
        token_embeddings = token_embeddings.reshape(b, t, self.config.n_embd)
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  
        x = self.drop(token_embeddings + position_embeddings)
        intermediate_outputs = list()
        for it in range(steps-1):
            x = self.blocks(x)
            if 1 and targets is not None:
                intermediate_outputs.append(self.head(self.ln_f(x))[:, :, self.mapping])
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        logits = logits[:, :, self.mapping]
        # if we are given some desired targets also calculate the loss
        loss = 0
        if targets is not None:
            for intermediate_output in intermediate_outputs:
                loss += F.cross_entropy(
                    intermediate_output.view(-1, intermediate_output.size(-1)),
                    targets.view(-1),
                )
            loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


from torchvision import datasets, transforms

train_dataset = datasets.MNIST(
    "../data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

test_dataset = datasets.MNIST(
    "../data",
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

# =============================================================================
# Mapping
# =============================================================================
mapping = torch.tensor(np.random.choice(20, 10, replace=False)).long()

# initialize a baby GPT model
mconf = GPTConfig(9, 1, n_layer=1, n_head=4, n_embd=128, mapping=mapping)
model = GPT(mconf)




# initialize a trainer instance and kick off training
tconf = TrainerConfig(
    max_epochs=50,
    batch_size=256,
    learning_rate=6e-4,
    lr_decay=False,
    num_workers=0,
    ckpt_path="MNIST_mapping_model.pt",
)
trainer = Trainer(model, train_dataset, test_dataset, tconf)
eval_list = trainer.train()
