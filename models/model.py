import torch
from torch import nn
from .reconstructor import Reconstructor
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.time_recon = Reconstructor(
            i_dim=args.time_recon.in_dim,
            h_dim=args.time_recon.n_hiddens,
            res_h_dim=args.time_recon.n_residual_hiddens,
            n_res_layers=args.time_recon.n_residual_layers,
            n_embeddings=args.time_recon.n_embeddings,
            embedding_dim=args.time_recon.embedding_dim,
            beta=args.beta,
            activation=args.time_recon.activation,
            patch_size=args.time_recon.patch_size,
            top_k=args.time_recon.top_k
        )
        self.freq_recon = Reconstructor(
            i_dim=args.freq_recon.in_dim,
            h_dim=args.freq_recon.n_hiddens,
            res_h_dim=args.freq_recon.n_residual_hiddens,
            n_res_layers=args.freq_recon.n_residual_layers,
            n_embeddings=args.freq_recon.n_embeddings,
            embedding_dim=args.freq_recon.embedding_dim,
            beta=args.beta,
            activation=args.freq_recon.activation,
            patch_size=args.freq_recon.patch_size,
            top_k=args.freq_recon.top_k
        )
        
        
    def time_reconstruct(self, x, mask=None):
        embedding_loss, x_hat, perplexity, min_encoding_indices = self.time_recon(x, mask)

        return embedding_loss, x_hat, perplexity, min_encoding_indices

    def to_freq(self, x):
        '''
        Transform time to freq

        Input:
        - x (b, l ,c)
        
        Output:
        - x_fft_cat (b, l, c)
        '''
        b,l,c = x.shape
        # (b, l ,c) -> (b, c, l)
        x_t = x.permute(0, 2, 1).contiguous()
        x_fft = torch.fft.rfft(x_t, dim=-1)
        x_fft_cat = torch.cat([x_fft.real[:,:,:], x_fft.imag[:,:,1:-1]], dim=-1)
        x_fft_cat = x_fft_cat.permute(0, 2, 1).contiguous()
        return x_fft_cat
    
    def freq_reconstruct(self, x, mask=None):
        
        b,l,c = x.shape
        # time to freq        
        # x -> (batch_size, in_dim, seq_len)
        x_t = x.permute(0, 2, 1).contiguous()
        x_fft = torch.fft.rfft(x_t, dim=-1)
        x_fft_amp = torch.abs(x_fft)
        x_fft_amp = x_fft_amp.permute(0, 2, 1).contiguous()
        
        embedding_loss, x_fft_amp_hat, perplexity, min_encoding_indices  = self.freq_recon(x_fft_amp, mask)

        # freq to time
        x_fft_amp_hat_ = x_fft_amp_hat.permute(0, 2, 1).contiguous()
        x_fft_hat_real = x_fft_amp_hat_[:, :, :x_fft.shape[-1]] * torch.cos(torch.angle(x_fft))
        x_fft_hat_imag = x_fft_amp_hat_[:, :, :x_fft.shape[-1]] * torch.sin(torch.angle(x_fft))
        x_hat = torch.fft.irfft(torch.complex(x_fft_hat_real, x_fft_hat_imag), n=l, dim=-1)
        x_hat = x_hat.permute(0, 2, 1).contiguous()
        return embedding_loss, x_hat, x_fft_amp_hat, perplexity, min_encoding_indices
    
        # b,l,c = x.shape
        # # time to freq
        
        # # x -> (batch_size, in_dim, seq_len)
        # x_t = x.permute(0, 2, 1).contiguous()
        # x_fft = torch.fft.rfft(x_t, dim=-1)
        # x_fft_cat = torch.cat([x_fft.real[:,:,:], x_fft.imag[:,:,1:-1]], dim=-1)
        # x_fft_cat = x_fft_cat.permute(0, 2, 1).contiguous()
        
        # embedding_loss, x_fft_hat, perplexity, min_encoding_indices  = self.freq_recon(x_fft_cat, mask)
        
        # # freq to time
        # x_fft_hat_ = x_fft_hat.permute(0, 2, 1).contiguous()
        # x_fft_hat_real = x_fft_hat_[:, :, :x_fft.shape[-1]]
        # x_fft_hat_imag = x_fft_hat_[:, :, x_fft.shape[-1]:]
        # x_fft_hat_imag = torch.cat([torch.zeros_like(x_fft.imag[:, :, 0:1]),x_fft_hat_imag,torch.zeros_like(x_fft.imag[:, :, -1:])],dim=-1)
        # x_hat = torch.fft.irfft(torch.complex(x_fft_hat_real, x_fft_hat_imag), n=l, dim=-1)
        # x_hat = x_hat.permute(0, 2, 1).contiguous()
        
        # return embedding_loss, x_hat, x_fft_hat, perplexity, min_encoding_indices


        
    def forward(self, x, mask=None, only_time=False, only_freq=False, indices=False):
        x_ = x    
        if only_time:
            assert mask is None or isinstance(mask, torch.Tensor) or (isinstance(mask, tuple) and len(mask) == 2,), f"Expected mask to be a torch.Tensor when only_time is True, got {type(mask)}"
            if isinstance(mask, tuple) and len(mask) == 2:
                mask, _ = mask
            embedding_loss1, x_hat1, perplexity1,min_encoding_indices1 = self.time_reconstruct(x_,mask)
            embedding_loss2, x_hat2, perplexity2,min_encoding_indices2 = torch.zeros_like(embedding_loss1), torch.zeros_like(x_hat1), torch.zeros_like(perplexity1), torch.zeros_like(min_encoding_indices1)
            x_hat = x_hat1
            embedding_loss = embedding_loss1
        elif only_freq:
            assert mask is None or isinstance(mask, torch.Tensor) or (isinstance(mask, tuple) and len(mask) == 2,), f"Expected mask to be a torch.Tensor when only_freq is True, got {type(mask)}"
            if isinstance(mask, tuple) and len(mask) == 2:
                _, mask = mask
            embedding_loss2, x_hat2, x_fft_amp_hat, perplexity2, min_encoding_indices2 = self.freq_reconstruct(x_,mask)
            embedding_loss1, x_hat1, perplexity1,min_encoding_indices1 = torch.zeros_like(embedding_loss2), torch.zeros_like(x_hat2), torch.zeros_like(perplexity2), torch.zeros_like(min_encoding_indices2)
            x_hat = x_hat2
            embedding_loss = embedding_loss2
        else: 
            
            if mask is not None:
                assert isinstance(mask, tuple) and len(mask) == 2, f"Expected mask to be a tuple of (time_mask, freq_mask), got {type(mask)} with length {len(mask) if isinstance(mask, tuple) else 'N/A'}"
                time_mask, freq_mask = mask
                embedding_loss1, x_hat1, perplexity1,min_encoding_indices1 = self.time_reconstruct(x_, time_mask)
                embedding_loss2, x_hat2, x_fft_amp_hat, perplexity2, min_encoding_indices2 = self.freq_reconstruct(x_, freq_mask)
            else:
                embedding_loss1, x_hat1, perplexity1,min_encoding_indices1 = self.time_reconstruct(x_, None)
                embedding_loss2, x_hat2, x_fft_hat, perplexity2, min_encoding_indices2 = self.freq_reconstruct(x_, None)

            x_hat = (x_hat1 + x_hat2) / 2
            embedding_loss = embedding_loss1 + embedding_loss2  #+ F.mse_loss(x_hat1, x_hat2)

        if indices:
            return embedding_loss, x_hat, perplexity1, perplexity2, min_encoding_indices1, min_encoding_indices2
        else:
            return embedding_loss, x_hat, perplexity1, perplexity2
    


if __name__ == "__main__":
    pass
