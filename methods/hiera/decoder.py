import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor

from hiera.layers import VitBlock


class Decoder(nn.Module):
    def __init__(self, dim, num_blocks, num_heads, use_bias=True):
        super().__init__()
        self.mask_embedding = nn.Parameter(torch.randn(dim))

        self.positional_embedding = nn.Parameter(
            torch.randn((1, 49, dim))
        )

        self.blocks = nn.ModuleList(
            [
                VitBlock(
                    in_dim=dim,
                    out_dim=dim,
                    mlp_multiplier=4,
                    num_heads=num_heads,
                    q_stride=1,
                    window_size=0,
                    use_mask_unit_attn=False,
                    use_bias=use_bias,
                ) for _ in range(num_blocks)
            ]
        )

        self.linear_out = nn.Linear(dim, 3 * 32 * 32, bias=use_bias)

        self.apply(self.init_weights)

    def forward(
        self,
        pred: Float[Tensor, "B nun_masked_tokens C"],
        mask: Bool[Tensor, "B num_windows"],
    ):
        B, N = mask.shape
        _, _, C = pred.shape

        empty_image = self.mask_embedding.to(dtype=pred.dtype).view(1, 1, C).expand(B, N, C).clone()
        empty_image[mask] = pred.reshape(-1, C)
        x = empty_image

        x = x + self.positional_embedding

        for block in self.blocks:
            x: Float[Tensor, "B N C"] = block(x)
        
        # --- 3 channels, 32 * 32 window size ---
        x: Float[Tensor, "B N 3*32*32"] = self.linear_out(x)
        x: Float[Tensor, "B 3 H W"] = rearrange(
            x, "B (H W) (P1 P2 C) -> B C (H P1) (W P2)", H=224 // 32, W=224 // 32, P1=32, P2=32
        )

        return x


    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
