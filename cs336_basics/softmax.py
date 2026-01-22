import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax values for each sets of scores in x along the specified dimension."""
    # x: (..., N, ...)
    # 为了数值稳定性，减去最大值
    x_max = torch.max(x, dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    sum_e_x = torch.sum(e_x, dim=dim, keepdim=True)
    return e_x / sum_e_x
