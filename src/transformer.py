import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        # dim=128,depth=12，heads=8，dim_head=64,mlp_dim=128
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads # 64 x 8
        self.heads = heads # 8
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads # n=50,h=8
        # self.to_qkv(x)得到的尺寸为[b,50,64x8x3],然后chunk成3份
        # 也就是说，qkv是一个三元tuple,每一份都是[b,50,64x8]的大小
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 把每一份从[b,50,64x8]变成[b,8,50,64]的形式
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
		# 这一步不太好理解，q和k都是[b,8,50,64]的形式，50理解为特征数量，64为特征变量
        # dots.shape=[b,8,50,50]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # 不考虑mask这一块的内容
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
		# 对[b,8,50,50]的最后一个维度做softmax
        attn = dots.softmax(dim=-1)

		# 这个attn就是计算出来的自注意力值，和v做点乘，out.shape=[b,8,50,64]
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # out.shape变成[b,50,8x64]
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out.shape重新变成[b,60,128]
        out =  self.to_out(out)
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
    # dim=128,fn=Attention/FeedForward
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
    # dim=128,hidden_dim=128
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
