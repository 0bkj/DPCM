
import torch
import torch.nn as nn
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
            top_k
        ):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = Encoder(patch_size, h_dim, n_res_layers, res_h_dim, activation)
        self.prototype_retrieval = PrototypeRetrieval(i_dim* n_embeddings, embedding_dim, beta,k=top_k)
        self.mask = torch.zeros(i_dim, i_dim * n_embeddings)  # (c, n_e)
        for i in range(i_dim):
            self.mask[i, i * n_embeddings:(i + 1) * n_embeddings] = 1.0
        self.n_embeddings = i_dim * n_embeddings
        self.decoder = Decoder(h_dim, patch_size, n_res_layers, res_h_dim, activation)

            

    def forward(self, x, mask=None, verbose=False):
        # x.shape (batch_size, seq_len, in_dim)
        b,t,c = x.shape

        x = einops.rearrange(x, 'b t c -> (b c) t')
        if t % self.patch_size != 0:
            pad_len = self.patch_size - (t % self.patch_size)
            x = torch.nn.functional.pad(x, (0, pad_len), mode='constant', value=0)
        x = einops.rearrange(x, 'b (n p) -> b n p', p=self.patch_size)
        z_e = self.encoder(x)
        z_e = einops.rearrange(z_e, 'b n h -> b h n')

        n = z_e.shape[2]
        masked = einops.rearrange(self.mask, 'c n -> 1 c 1 n').expand(b, -1, n, -1).reshape(b * c * n, -1).to(dtype=torch.bool,device=z_e.device)
        embedding_loss, z_q, perplexity, indices = self.prototype_retrieval(z_e, masked)
            
        z_q = einops.rearrange(z_q, 'b h n -> b n h')
        z_q = self.decoder(z_q)
        z_q = einops.rearrange(z_q, 'b n p -> b (n p)')
        if t % self.patch_size != 0:
            z_q = z_q[:, :-pad_len]
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

    