import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from hiera.layers.vit_block import VitBlock


class Hiera(nn.Module):
    def __init__(
        self, dim: int, mlp_multiplier: int, num_heads: int, use_te=False, use_bias=True
    ):
        pass

    def forward(self, x: Float[Tensor, "B N C"]) -> Float[Tensor, "B N C"]:
        return x