import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor


class AttentiveProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()

        # this is fixes as per https://arxiv.org/abs/2502.08769
        self.head_dim = 64
        assert dim % self.head_dim == 0, "dim needs to be divisible by 64"
        self.num_heads = dim // self.head_dim

        self.query_token = nn.Parameter(torch.randn(dim) * 0.02)

        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.linear_out = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)

    def forward(self, x: Float[Tensor, "B N C"]):
        B = x.shape[0]
        x = rearrange(
            self.kv_proj(x),
            "B N (two num_heads head_dim) -> two B num_heads N head_dim",
            two=2,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        k, v = x[0], x[1]

        q = repeat(
            self.query_token,
            "C -> B C",
            B=B,
        )
        q = rearrange(
            q,
            "B (num_heads head_dim) -> B num_heads 1 head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        x: Float[Tensor, "B num_heads 1 head_dim"] = F.scaled_dot_product_attention(query=q, key=k, value=v)

        x = rearrange(
            x,
            "B num_heads 1 head_dim -> B (num_heads head_dim)",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        return self.linear_out(x)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
