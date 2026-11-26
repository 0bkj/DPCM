
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.encoder import Encoder
from models.prototype_retrieval import PrototypeRetrieval
from models.decoder import Decoder
import einops

class Reconstructor(nn.Module):
    def __init__(
            self,
            i_dim, h_dim, res_h_dim, n_res_layers,
            n_embeddings, embedding_dim, beta, 
            activation,
            patch_size,
            top_k,
            patch_stride,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        #     )
        self.encoder = Encoder(patch_size, h_dim, n_res_layers, res_h_dim, activation)
        self.proj1 = nn.Linear(h_dim, embedding_dim)
        self.prototype_retrieval = PrototypeRetrieval(i_dim* n_embeddings, embedding_dim, beta,k=top_k)
        self.n_variables = i_dim
        self.n_embeddings = i_dim * n_embeddings
        self.proj2 = nn.Linear(embedding_dim, h_dim)
        self.decoder = Decoder(h_dim, patch_size, n_res_layers, res_h_dim, activation)

            

    def forward(self, x, mask=None, verbose=False):
        # x.shape (batch_size, seq_len, in_dim)
        b,t,c = x.shape

        x = einops.rearrange(x, 'b t c -> (b c) t')

        # Right-pad to complete the final patch.
        if t <= self.patch_size:
            padded_length = self.patch_size
        else:
            num_patches = (t - self.patch_size + self.patch_stride - 1) // self.patch_stride + 1
            padded_length = (num_patches - 1) * self.patch_stride + self.patch_size
        pad_len = padded_length - t
        if pad_len:
            x = F.pad(x, (0, pad_len), mode='constant', value=0)

        # [batch * channels, latent_time, patch_size]
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride).contiguous()
        z_e = self.encoder(x)
        z_e = self.proj1(z_e)
        z_e = einops.rearrange(z_e, 'b n h -> b h n')

        if c != self.n_variables:
            raise ValueError(f"Expected {self.n_variables} variables, got {c}")
        embedding_loss, z_q, perplexity, indices = self.prototype_retrieval(
            z_e, num_variables=c
        )
            
        z_q = einops.rearrange(z_q, 'b h n -> b n h')
        z_q = self.proj2(z_q)
        z_q = self.decoder(z_q)
        # Average values where decoded patches overlap.
        z_q = F.fold(
            z_q.transpose(1, 2).contiguous(),
            output_size=(1, padded_length),
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_stride),
        ).squeeze(2).squeeze(1)
        overlap_count = F.fold(
            torch.ones(
                (z_e.shape[0], self.patch_size, z_e.shape[-1]),
                dtype=z_q.dtype,
                device=z_q.device,
            ),
            output_size=(1, padded_length),
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_stride),
        ).squeeze(2).squeeze(1)
        z_q = z_q / overlap_count.clamp_min(1.0)
        z_q = z_q[:, :t]
        x_hat = einops.rearrange(z_q, '(b c) t -> b t c', b=b, c=c)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, indices 
    
    
    def decode_indices(self, indices, verbose=False):
        # indices.shape (batch_size, seq_len)
        b,t = indices.shape
        
        z_q = self.prototype_retrieval.embedding(indices) # (batch_size, seq_len, embedding_dim)
        x_hat = self.decoder(z_q)
        
        if verbose:
            print('recon data shape:', x_hat.shape)
            assert False

        return x_hat

    
