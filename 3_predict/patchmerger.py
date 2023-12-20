import torch
from torch import nn
from einops import rearrange


class PatchMergerBottleneck(nn.Module):
    def __init__(self, dim, num_out_tokens, feat_drop=0., dim_proj=None, eps=1e-8, bottleneck=1024):
        super().__init__()
        
        if dim > bottleneck:
            self.bottleneck = nn.Linear(dim, bottleneck)
            dim = bottleneck
        else:
            self.bottleneck = nn.Identity()
        k = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.W = nn.Parameter(torch.empty(dim, num_out_tokens).uniform_(-k, k))
        self.W_scale = nn.Parameter(torch.ones(num_out_tokens))
        self.bias = nn.Parameter(torch.zeros(num_out_tokens))
        
        dim_proj = dim_proj or dim
        self.scale = dim_proj ** -0.5
        self.register_buffer("eps", torch.tensor(eps))
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim_proj))

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        Stolen from MAE.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked
        
    def forward(self, x):
        x = self.bottleneck(x)
        x = self.norm(x)
        x = self.feat_drop(x)
        
        W = self.W_scale * self.W / (self.W.norm(dim=0) + self.eps)
        a = self.scale * (torch.matmul(x, W) + self.bias)
        a = rearrange(a, "B N M -> B M N")
        a = a.softmax(dim=1)
        x = torch.matmul(a, x)
        
        x = self.proj(x)
        
        return x
