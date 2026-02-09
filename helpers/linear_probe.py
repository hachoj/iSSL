import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class LinearProbe(nn.Module):
    def __init__(self):
        pass

    def forward(self, x: Float[Tensor, "B N C"]) -> Tensor:
        return x