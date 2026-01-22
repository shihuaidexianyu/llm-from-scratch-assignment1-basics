import math

import torch
import torch.nn as nn
from torch import Tensor


class MyLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        no bias linear layer
        """
        super().__init__()

        def _init_parameter(tensor: Tensor) -> None:
            std = math.sqrt(2.0 / (in_features + out_features))
            nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        _init_parameter(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.weight.t()
