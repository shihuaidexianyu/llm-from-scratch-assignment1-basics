"""Given a tensor of inputs and targets, compute the average cross-entropy
loss across examples.

Args:
    inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
        unnormalized logit of jth class for the ith example.
    targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
        Each value must be between 0 and `num_classes - 1`.

Returns:
    Float[Tensor, ""]: The average cross-entropy loss across examples.
"""

import torch


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    # 1. 计算 LogSumExp (第一项)
    # shape: (batch_size, )
    log_sum_exp = torch.logsumexp(inputs, dim=-1)

    # 2. 取出正确答案对应的原始分数 (Target Logits)
    # inputs[i, targets[i]] 表示取第 i 个样本中，targets[i] 那个位置的分数
    # shape: (batch_size, )
    target_logits = inputs[torch.arange(inputs.size(0)), targets]

    # 3. 计算每个样本的 Loss
    # 公式: Loss = LogSumExp - Target_Logit
    loss_per_sample = log_sum_exp - target_logits

    # 4. 求平均值返回 (变成标量)
    return loss_per_sample.mean()
