import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


class AttentionBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
        use_bias=True,
    ):
        super().__init__()
        assert out_dim % num_heads == 0
        head_dim = out_dim // num_heads
        num_heads = num_heads

        self.project_kqv = nn.Linear(in_dim, 3 * out_dim, bias=use_bias)   

        self.proj = nn.Linear(out_dim, out_dim, bias=use_bias)

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.q_stride = q_stride
        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

        self.apply(self._init_weights)

    def forward(self, x: Float[Tensor, "B N C"]) -> Float[Tensor, "B N C"]:
        B, N, _ = x.shape

        num_windows = (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1

        x: Float[Tensor, "B N 3C"] = self.project_kqv(x)

        x: Float[Tensor, "3 B H num_windows N head_dim"] = (
            x.reshape(B, -1, num_windows, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = x[0], x[1], x[2]

        if self.q_stride > 1:
            # --- This only works because of the Unroll operations performed 
            # after adding the positional embedding ---
            q = (
                q.view(B, self.num_heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        # with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        #     x = F.scaled_dot_product_attention(q, k, v)
        x: Float[Tensor, "B H num_windows N head_dim"] = F.scaled_dot_product_attention(q, k, v)

        x: Float[Tensor, "B N C"] = x.transpose(1, 3).reshape(B, -1, self.num_heads * self.head_dim)
        
        x: Float[Tensor, "B N C"] = self.proj(x)        

        return x

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
