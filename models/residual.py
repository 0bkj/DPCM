
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualMLP(nn.Module):
    def __init__(self, h_dim, res_h_dim, activation=None):
        super(ResidualMLP, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(h_dim, res_h_dim),
            activation(),
            nn.Linear(res_h_dim, h_dim),
        )

    def forward(self, x):
        return x + self.block(x)


if __name__ == "__main__":
    # random data
    pass
