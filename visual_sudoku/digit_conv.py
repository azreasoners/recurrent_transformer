import torch.nn as nn
from torch.nn import functional as F

class DigitConv(nn.Module):
    """
    Convolutional neural network for MNIST digit recognition.
    Slightly adjusted from SATNet repository
    """

    def __init__(self, config):
        super(DigitConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # self.fc2 = nn.Linear(500, 10)
        self.fc2 = nn.Linear(500, config.n_embd)

    def forward(self, x):
        batch_size, block_size = x.shape[0], x.shape[1]
        x = x.view(-1, 1, 28, 28) # (batch_size * block_size, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return F.softmax(x, dim=1)[:, :9].contiguous()
        return x.view(batch_size, block_size, -1)
