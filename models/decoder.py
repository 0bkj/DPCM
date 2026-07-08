import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualMLP
import einops

class Decoder(nn.Module):
    """
    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, h_dim, out_dim, n_res_layers, res_h_dim, activation=None):
        super(Decoder, self).__init__()
        if activation == "silu":
            activation = nn.SiLU
        elif activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
            
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            activation(),
            ResidualMLP(h_dim, res_h_dim, activation),
            activation(),
            nn.Linear(h_dim, out_dim),
        )

    def forward(self, x):
        # x (b, n, h)
        x = self.mlp(x)
        return x



if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 64, 10))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(64, 128, 40, 3, 64)
    decoder_out = decoder(x)
    print("Decoder out shape:", decoder_out.shape)

    decoder_out = decoder(x)
    print("PatchDecoder out shape:", decoder_out.shape)
