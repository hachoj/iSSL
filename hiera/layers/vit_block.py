import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from einops import rearrange

from .ffn import FFN
from .attention_block import AttentionBlock

from torch import Tensor
from jaxtyping import Float


class VitBlock(nn.Module):
    def __init__(self, dim: int, mlp_multiplier:int, num_heads: int, use_te=False, use_bias=True):
        self.ffn = FFN(dim=dim, mlp_multiplier=mlp_multiplier, use_te=use_te, use_bias=use_bias)
        self.attention = AttentionBlock(dim=dim, num_heads=num_heads, use_te=use_te, use_bias=use_bias)

    def forward(self, x: Float[Tensor, "B N C"]) -> Float[Tensor, "B N C"]:
        x = self.attention(x)
        x = self.ffn(x)
        return x