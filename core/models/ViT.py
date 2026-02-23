from typing import Tuple

import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from torch import Tensor

from core.layers import Patchify, VitBlock


class ViT(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_blocks: int,
        mlp_ratio: float,
        input_size: Tuple[int, int] = (224, 224),
        patch_kernel: Tuple[int, int] = (7, 7),
        patch_stride: Tuple[int, int] = (4, 4),
        patch_padding: Tuple[int, int] = (3, 3),
        drop_path_p: float = 0.0,
        norm: str = "layernorm",
        use_swiglu: bool = False,
        use_bias=True,
    ):
        super().__init__()
        assert norm in ["layernorm", "rmsnorm"]
        self.num_patches = (input_size[0] // patch_stride[0]) * (input_size[1] // patch_stride[1])
        self.patchify = Patchify(
            in_channels=3,
            out_channels=dim,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            use_bias=use_bias,
        )

        self.positional_embedding = nn.Parameter(
            torch.randn((1, self.num_patches, dim))
        )

        self.blocks = nn.ModuleList(
            [
                VitBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    drop_path_p=drop_path_p,
                    norm=norm,
                    use_swiglu=use_swiglu,
                    use_bias=use_bias,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Float[Tensor, "B C H W"], mask_ratio: float | None = None):
        if mask_ratio is not None:
            x, mask = self.masked_forward(x, mask_ratio=mask_ratio)
        else:
            x = self.full_forward(x)
            mask = None

        return x, mask

    def masked_forward(self, x: Float[Tensor, "B C H W"], mask_ratio: float):
        x: Float[Tensor, "B N C"] = self.patchify(x)
        x: Float[Tensor, "B N C"] = x + self.positional_embedding

        x, mask = self.mask_tokens(x, mask_ratio=mask_ratio)

        for block in self.blocks:
            x = block(x)

        return x, mask

    def full_forward(self, x: Float[Tensor, "B C H W"]):
        x: Float[Tensor, "B N C"] = self.patchify(x)
        x: Float[Tensor, "B N C"] = x + self.positional_embedding

        for block in self.blocks:
            x = block(x)

        return x

    def mask_tokens(
        self, x: Float[Tensor, "B N C"], mask_ratio: float
    ) -> tuple[Float[Tensor, "B L C"], Bool[Tensor, "B N"]]:
        if not (0.0 <= mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in [0.0, 1.0), got {mask_ratio}.")

        B, N, C = x.shape
        num_keep = max(1, int(N * (1.0 - mask_ratio)))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        ids_keep, _ = ids_keep.sort(dim=1)

        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, C),
        )

        # mask=True means masked (removed), mask=False means visible (kept).
        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return x_masked, mask
