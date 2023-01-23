import torch.nn as nn

class NonogramEmb(nn.Module):
    def __init__(self, config):
        super(NonogramEmb, self).__init__()
        self.emb = nn.Embedding(
            config.max_hint_value + 1,
            config.n_embd // (2 * config.max_num_per_hint)
            )
    def forward(self, x):
        batch_size, block_size = x.shape[0], x.shape[1]
        return self.emb(x).view(batch_size, block_size, -1)
