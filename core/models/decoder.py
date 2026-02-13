import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor

from core.layers import VitBlock


class Decoder(nn.Module):
    def __init__(
        self,
        prev_dim: int,
        dim: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        input_size: Tuple[int, int],
        patch_stride: Tuple[int, int],
        use_swiglu: bool = False,
        use_bias=True,
    ):
        super().__init__()
        self.num_patches = (input_size[0] // patch_stride[0]) * (
            input_size[1] // patch_stride[1]
        )
        self.dim = dim
        self.input_size = input_size
        self.patch_stride = patch_stride

        self.mask_embedding = nn.Parameter(torch.randn(dim))

        self.positional_embedding = nn.Parameter(
            torch.randn((1, self.num_patches, dim))
        )

        self.input_proj = nn.Linear(
            in_features=prev_dim,
            out_features=dim,
            bias=use_bias
        )

        self.blocks = nn.ModuleList(
            [
                VitBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    drop_path_p=0.0,
                    use_swiglu=use_swiglu,
                    use_bias=use_bias,
                )
                for _ in range(num_blocks)
            ]
        )

        self.linear_out = nn.Linear(dim, 3 * math.prod(patch_stride), bias=use_bias)

        self.apply(self._init_weights)

    def forward(
        self,
        pred: Float[Tensor, "B num_visible_tokens C"],
        mask: Bool[Tensor, "B num_patches"],
    ):
        B, N = mask.shape
        pred = self.input_proj(pred)

        empty_image = (
            self.mask_embedding.to(dtype=pred.dtype)
            .view(1, 1, self.dim)
            .expand(B, N, self.dim)
            .clone()
        )

        visible_mask = ~mask
        empty_image[visible_mask] = pred.reshape(-1, self.dim)
        x = empty_image

        x = x + self.positional_embedding

        for block in self.blocks:
            x: Float[Tensor, "B N C"] = block(x)

        # --- 3 channels, p * p patch size ---
        p1, p2 = self.patch_stride
        h, w = self.input_size[0] // p1, self.input_size[1] // p2

        x: Float[Tensor, "B N 3*p*p"] = self.linear_out(x)
        x: Float[Tensor, "B 3 H W"] = rearrange(
            x,
            "B (h w) (p1 p2 C) -> B C (h p1) (w p2)",
            h=h,
            w=w,
            p1=p1,
            p2=p2,
        )

        return x

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
