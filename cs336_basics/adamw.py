from collections.abc import Callable
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}

        # 1. 调用父类初始化，它会处理 params 生成器并将其存入 self.param_groups
        # 注意：这里不要初始化 self.m 或 self.v，因为每个参数都需要独立的 m 和 v
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # 2. 获取该参数对应的梯度
                grad = p.grad.data

                # 3. 获取该参数对应的状态字典 (State Dictionary)
                state = self.state[p]

                # 4. 懒加载初始化：如果是第一次更新该参数，初始化 m 和 v
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)  # 对应你的 self.m
                    state["exp_avg_sq"] = torch.zeros_like(p)  # 对应你的 self.v

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # 5. 更新一阶矩 (m) 和 二阶矩 (v)
                # 使用 in-place 操作 (.mul_, .add_) 以节省内存
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 6. 计算偏差修正后的步长 (Alpha_t)
                # alpha_t = lr * (sqrt(1 - beta2^t) / (1 - beta1^t))
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)

                # 7. 更新参数 (应用 Adam 更新规则)
                # p = p - step_size * m / (sqrt(v) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # 8. 应用权重衰减 (Weight Decay)
                # p = p - lr * weight_decay * p
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss
