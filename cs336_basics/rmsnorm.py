import torch
import torch.nn as nn
from torch import Tensor


class MyRMSNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        RMSNorm layer
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones((dim,), device=device, dtype=dtype))  # 参数向量, 初始化为1

    def forward(self, input: Tensor) -> Tensor:
        input = input.to(torch.float32)  # 确保计算时使用float32
        rms = input.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()  # 计算RMS
        return input / rms * self.scale  # 逐位相乘
