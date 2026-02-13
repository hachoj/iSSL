import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from hiera.helpers import do_pool

from .attention_block import AttentionBlock
from .drop_path import DropPath
from .ffn import FFN


class VitBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mlp_multiplier: int,
        num_heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
        drop_path_p: float = 0.0,
        use_bias=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.q_stride = q_stride
        self.drop_path = DropPath(drop_path_p)

        self.norm1 = nn.LayerNorm(in_dim)
        self.attention = AttentionBlock(
            in_dim=in_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            q_stride=q_stride,
            window_size=window_size,
            use_mask_unit_attn=use_mask_unit_attn,
            use_bias=use_bias,
        )
        self.norm2 = nn.LayerNorm(out_dim)
        self.ffn = FFN(
            dim=out_dim,
            mlp_multiplier=mlp_multiplier,
            use_bias=use_bias,
        )
        self.proj = (
            nn.Linear(
                in_features=in_dim,
                out_features=out_dim,
                bias=use_bias,
            )
            if in_dim != out_dim
            else None
        )

    def forward(self, x: Float[Tensor, "B N C"]) -> Float[Tensor, "B N C"]:
        x_norm = self.norm1(x)
        if self.proj is not None:
            x = do_pool(self.proj(x_norm), stride=self.q_stride)
        x = x + self.drop_path(self.attention(x_norm))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
