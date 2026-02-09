import torch
import torch.nn as nn

from einops import rearrange

from torch import Tensor
from jaxtyping import Float


class Patchify(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, use_bias=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(patch_size, patch_size),
            padding=0,
            stride=(patch_size, patch_size),
            bias=use_bias,
        )

        self.apply(self._init_weights)

    def forward(self, x: Float[Tensor, "B C H W"]):
        x: Float[Tensor, "B C' H' W'"] = self.conv(x)
        B, C, H, W = x.shape
        x: Float[Tensor, "B N C'"] = rearrange(x, "B C' H' W' -> B (H' W') C'")
        return x

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
