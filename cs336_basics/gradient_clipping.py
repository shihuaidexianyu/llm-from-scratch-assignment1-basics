from collections.abc import Iterable
import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    gradients = [p.grad for p in parameters if p.grad is not None]
    # 计算所有梯度的平方和
    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in gradients]), 2)

    # 计算缩放因子
    # 使用 torch.clamp 避免 if/else，并保持计算图连续性（如果需要）
    clip_coef = max_l2_norm / (total_norm + 1e-6)

    # 原地修改梯度
    if total_norm > max_l2_norm:
        for grad in gradients:
            grad.mul_(clip_coef)
