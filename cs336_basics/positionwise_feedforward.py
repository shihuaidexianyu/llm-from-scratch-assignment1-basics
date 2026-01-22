import torch
import torch.nn as nn
from torch import Tensor
from .linear import MyLinear


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class MyPositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Position-wise Feed-Forward Network
        """
        real_hidden_dim = hidden_dim // 3  # GELU 推荐的隐藏层维度
        super().__init__()
        self.W1 = MyLinear(dim, real_hidden_dim, device=device, dtype=dtype)
        self.W2 = MyLinear(real_hidden_dim, dim, device=device, dtype=dtype)
        self.W3 = MyLinear(dim, real_hidden_dim, device=device, dtype=dtype)
        self.activation = swish

    def forward(self, input: Tensor) -> Tensor:
        signal = self.activation(self.W1(input))
        gate = self.W3(input)
        return self.W2(signal * gate)
