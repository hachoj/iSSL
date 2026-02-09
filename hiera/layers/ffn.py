import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class FFN(nn.Module):
    def __init__(self, dim: int, mlp_multiplier: int, use_bias=True):
        hidden_dim: int = int(dim * mlp_multiplier)

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.ModuleList(
            [
                nn.Linear(in_features=dim, out_features=hidden_dim, bias=use_bias),
                nn.GELU(),
                nn.Linear(in_features=hidden_dim, out_features=dim, bias=use_bias),
            ]
        )

        self.use_te = use_te

        self.apply(self._init_weights)

    def forward(self, x: Float[Tensor, "B N C"]) -> Float[Tensor, "B N C"]:
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        return residual + x

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)