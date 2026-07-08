
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualMLP
import einops

class Encoder(nn.Module):
    """
    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, activation=None):
        super(Encoder, self).__init__()
        if activation == "silu":
            activation = nn.SiLU
        elif activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        # in_dim = window_size // patch_size
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            activation(),
            ResidualMLP(h_dim, res_h_dim, activation),
            activation(),
            nn.Linear(h_dim, h_dim),
            activation(),    
        )

    def forward(self, x):
        #  x (b, n, p)
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((32, 100, 40))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(100, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
    
    
    
    
