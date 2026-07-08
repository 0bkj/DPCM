import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

class PrototypeRetrieval(nn.Module):
    """
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    - k : number of top embeddings to select (default: 3 for Top-K VQ)
    - tau : temperature for softmax
    """

    def __init__(self, n_e, e_dim, beta, k=3, tau=0.3):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.k = k
        self.tau = tau

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, masked=None):
        """
        Inputs:
            z: (batch, channel, length)
            masked: None, or (batch*length, n_e) for per-sample mask
        """
        # reshape z -> (batch, length, channel) and flatten
        b, c, l = z.shape
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        device = z.device
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        if masked is not None:
            assert masked.shape == (z_flattened.shape[0], self.n_e), f"masked shape {masked.shape} != {(z_flattened.shape[0], self.n_e)}"
            # Per-sample mask: mask out disallowed embeddings by setting distance to inf
            d = d + (~masked) * 1e10
        
        # Soft Top-K selection with straight-through estimator
        # Calculate softmax weights
        soft_weights = F.softmax(-d / self.tau, dim=1)
        _, topk_indices = torch.topk(soft_weights, self.k, dim=1)
        min_encodings_hard = torch.zeros(z_flattened.shape[0], self.n_e).to(device)
        min_encodings_hard.scatter_(1, topk_indices, 1.0)
        topk_mask = torch.zeros_like(soft_weights)
        topk_mask.scatter_(1, topk_indices, 1.0)
        soft_weights_masked = soft_weights * topk_mask
        soft_weights_normalized = soft_weights_masked / (soft_weights_masked.sum(dim=1, keepdim=True) + 1e-10)
        min_encodings = soft_weights_normalized
        
        min_encoding_indices = topk_indices
        min_encoding_indices = einops.rearrange(min_encoding_indices, '(b l) k -> b l k', b=b, l=l)
  
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 2, 1).contiguous()

        return loss, z_q, perplexity, min_encoding_indices
