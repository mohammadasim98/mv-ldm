# MIT License

# Copyright (c) 2022 Karl Stelzner

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file comes from https://github.com/stelzner/srt

import torch
from einops import rearrange
from torch import nn
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import torch.nn.functional as F
# from xformers.components.attention import ScaledDotProduct
from torch.nn.attention import SDPBackend, sdpa_kernel

def generate_atten_mask(L, S):
    return torch.triu(torch.ones(L, S) * float('-inf'), diagonal=1)

class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, selfatt=True, kv_dim=None, flash_atten=False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.flash_atten = flash_atten
        self.attend = nn.Softmax(dim=-1)
        # self.attention = ScaledDotProduct().cuda()
        if selfatt:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
    # @torch.compile()
    def forward(self, x, z=None, mask=None):
        if z is None:
            B, L, *_ = x.shape
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            S = L

        else:
            B, L, *_ = x.shape
            B, S, *_ = z.shape
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        # out = flash_attn_func(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16))
        # out = out[:, 0:1, ...].to(torch.float32)
        attn_mask = None
        if mask is not None:
            # attn_mask = generate_atten_mask(L, S)
            # attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)
            mask = mask.float()
            mask = rearrange(mask, "B L S-> B () L S")
            attn_mask = mask.expand(-1, self.heads, -1, -1)
            # mask2 = mask1.transpose(1, 2)
            # attn_mask = mask1 @ mask2
            
            # NOTE: Setting the min value to -inf can cause nan during sampling when there are groups that contains only padding values 
            # attn_mask = torch.where(attn_mask > 0, 0.0, float('-inf'))
            
            # HACK: Instead of -inf, set it to some smallest value of the range of the datatype to avoid nan
            attn_mask = torch.where(attn_mask > 0, 0.0, float("-inf"))

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            # with torch.autocast(device_type="cuda"):

            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        # out = self.attention(q=q, k=k, v=v, mask=attn_mask)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out.to(torch.float32))