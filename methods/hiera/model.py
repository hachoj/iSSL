# TODO: FIGURE OUT WHAT INIT FOR LAYERNORM

import math
from typing import Tuple

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor

from hiera.helpers import Unroll  # , Reroll
from hiera.layers import Patchify, VitBlock


class Hiera(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, ...],
        blocks: Tuple[int, ...],
        heads: Tuple[int, ...],
        mlp_multiplier: int,
        window_size: int,  # in tokens, not pixels
        input_size: Tuple[int, int] = (224, 224),
        patch_kernel: Tuple[int, int] = (7, 7),
        patch_stride: Tuple[int, int] = (4, 4),
        patch_padding: Tuple[int, int] = (3, 3),
        q_stride: Tuple[int, int] = (2, 2),
        drop_path_p: float = 0.0,
        use_bias=True,
    ):
        super().__init__()
        assert len(dims) == len(blocks) and len(dims) == len(heads)
        """
        Hiera-B: channels [96, 192, 384, 768], blocks [2, 3, 16, 3], heads [1, 2, 4, 8], 9G FLOPs, 52M parameters.
        """
        self.window_size = window_size

        self.patchify = Patchify(
            in_channels=3,
            out_channels=dims[0],
            patch_kernel=patch_kernel,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            use_bias=use_bias,
        )

        self.positional_embedding = nn.Parameter(
            torch.randn((1, (input_size[0] // patch_stride[0])**2, dims[0]))
        )

        self.unroll = Unroll(
            input_size, patch_stride, [q_stride] * 3
        )

        self.blocks = nn.ModuleList()

        # --- Construct Hierarchical Blocks ---
        # -- stage 1 --
        dim, num_blocks, num_heads = dims[0], blocks[0], heads[0]
        for _ in range(num_blocks):
            self.blocks.append(
                VitBlock(
                    in_dim=dim,
                    out_dim=dim,
                    mlp_multiplier=mlp_multiplier,
                    num_heads=num_heads,
                    q_stride=1,
                    window_size=window_size,
                    use_mask_unit_attn=True,
                    use_bias=use_bias,
                    drop_path_p=drop_path_p,
                )
            )
        prev_dim = dim

        # -- stage 2 --
        dim, num_blocks, num_heads = dims[1], blocks[1], heads[1]
        # -- stage 2 pool --
        self.blocks.append(
            VitBlock(
                in_dim=prev_dim,
                out_dim=dim,
                mlp_multiplier=mlp_multiplier,
                num_heads=num_heads,
                q_stride=math.prod(q_stride),  # 4 = 2x2
                window_size=window_size // math.prod(q_stride),
                use_mask_unit_attn=True,
                drop_path_p=0.0,
                use_bias=use_bias,
            )
        )
        for _ in range(num_blocks - 1):
            self.blocks.append(
                VitBlock(
                    in_dim=dim,
                    out_dim=dim,
                    mlp_multiplier=mlp_multiplier,
                    num_heads=num_heads,
                    q_stride=1,
                    window_size=window_size // math.prod(q_stride),
                    use_mask_unit_attn=True,
                    drop_path_p=drop_path_p,
                    use_bias=use_bias,
                )
            )
        prev_dim = dim

        # -- stage 3 --
        dim, num_blocks, num_heads = dims[2], blocks[2], heads[2]
        # -- stage 3 pool --
        self.blocks.append(
            VitBlock(
                in_dim=prev_dim,
                out_dim=dim,
                mlp_multiplier=mlp_multiplier,
                num_heads=num_heads,
                q_stride=math.prod(q_stride),  # 4 = 2x2
                window_size=window_size // (math.prod(q_stride) ** 2),
                use_mask_unit_attn=True,
                drop_path_p=0.0,
                use_bias=use_bias,
            )
        )
        for _ in range(num_blocks - 1):
            self.blocks.append(
                VitBlock(
                    in_dim=dim,
                    out_dim=dim,
                    mlp_multiplier=mlp_multiplier,
                    num_heads=num_heads,
                    q_stride=1,
                    window_size=0,
                    use_mask_unit_attn=False,
                    drop_path_p=drop_path_p,
                    use_bias=use_bias,
                )
            )
        prev_dim = dim

        # -- stage 4 --
        dim, num_blocks, num_heads = dims[3], blocks[3], heads[3]
        # -- stage 4 pool --
        self.blocks.append(
            VitBlock(
                in_dim=prev_dim,
                out_dim=dim,
                mlp_multiplier=mlp_multiplier,
                num_heads=num_heads,
                q_stride=math.prod(q_stride),  # 4 = 2x2
                window_size=0,
                use_mask_unit_attn=False,
                drop_path_p=0.0,
                use_bias=use_bias,
            )
        )
        for _ in range(num_blocks - 1):
            self.blocks.append(
                VitBlock(
                    in_dim=dim,
                    out_dim=dim,
                    mlp_multiplier=mlp_multiplier,
                    num_heads=num_heads,
                    q_stride=1,
                    window_size=0,
                    use_mask_unit_attn=False,
                    drop_path_p=drop_path_p,
                    use_bias=use_bias,
                )
            )

    def forward(self, x: Float[Tensor, "B C H W"], mask_ratio: float | None = None):
        if mask_ratio is not None:
            x_masked, mask = self.masked_forward(x, mask_ratio=mask_ratio)
        else:
            x_masked = self.full_forward(x)
            mask = None

        return x_masked, mask
    
    def masked_forward(self, x: Float[Tensor, "B C H W"], mask_ratio: float = 0.6):
        x: Float[Tensor, "B N C"] = self.patchify(x)
        x: Float[Tensor, "B N C"] = x + self.positional_embedding
        x: Float[Tensor, "B N C"] = self.unroll(x)

        x_masked, mask = self.mask_mask_units(x, mask_ratio=mask_ratio)

        for block in self.blocks:
            x_masked = block(x_masked)

        return x_masked, mask
    
    def full_forward(self, x: Float[Tensor, "B C H W"]):
        x: Float[Tensor, "B N C"] = self.patchify(x)
        x: Float[Tensor, "B N C"] = x + self.positional_embedding
        x: Float[Tensor, "B N C"] = self.unroll(x)

        for block in self.blocks:
            x = block(x)

        return x

    def mask_mask_units(self, x: Float[Tensor, "B N C"], mask_ratio):
        B, N, C = x.shape
        num_windows = N // self.window_size
        
        num_keep = int(num_windows * (1 - mask_ratio))
        
        noise = torch.rand(B, num_windows, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]  # (B, num_keep)
        ids_keep, _ = ids_keep.sort(dim=1)
        
        ids_keep_expanded = ids_keep.unsqueeze(-1) * self.window_size + torch.arange(self.window_size, device=x.device)  # (B, num_keep, window_size)
        ids_keep_expanded = ids_keep_expanded.reshape(B, -1)  # (B, num_keep * window_size)
        
        x_masked = torch.gather(x, 1, ids_keep_expanded.unsqueeze(-1).expand(-1, -1, C))
        
        mask = torch.zeros(B, num_windows, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, True)
        
        return x_masked, mask
