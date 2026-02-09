import torch
import torch.nn as nn
import torch.nn.function as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


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
    """
    Args:
        q_stride: if greater than 1, pool q with this stride.
        window_size: 
    """
        super().__init__()
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        num_heads = num_heads

        self.norm = nn.LayerNorm(dim)

        self.project_kqv = nn.Linear(in_dim, 3 * out_dim, bias=use_bias)   

        self.proj = nn.Linear(dim, dim, bias=use_bias)

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.use_te = use_te
        self.q_stride = q_stride
        self.window_size = window_size
        self.use_mask_unit_attention = use_mask_unit_attention

        self.apply(self._init_weights)

    def forward(self, x: Float[Tensor, "B N C"]) -> Float[Tensor, "B N C"]:
        B, N, C = x.shape
        residual = x

        num_windows = (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attention else 1

        x = self.norm(x)

        x = rearrange(
            self.project_kqv(x),
            "B N (three head_dim) -> three B H num_windows N head_dim",
            three=3,
            H=self.num_heads,
            head_dim=self.head_dim,
            num_windows=num_windows,
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

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            x = F.scaled_dot_product_attention(q, k ,v)

        x = self.proj(x)        

        return residual + x

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)