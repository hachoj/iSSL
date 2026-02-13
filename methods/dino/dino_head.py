import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te


class DINO_Head(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def init_weights(module: nn.Module, name: str = ""):
        if isinstance(module, te.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)  # pyrefly:ignore
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # pyrefly:ignore
