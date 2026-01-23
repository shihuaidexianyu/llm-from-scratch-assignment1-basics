import torch

from collections.abc import Callable
import math


def set_seed(seed=42):
    torch.manual_seed(seed)  # CPU 随机数
    torch.cuda.manual_seed(seed)  # GPU 随机数 (如果有)


set_seed()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
        for p in group["params"]:
            if p.grad is None:
                continue
        state = self.state[p]  # Get state associated with p.
        t = state.get("t", 0)  # Get iteration number from the state, or initial value.
        grad = p.grad.data  # Get the gradient of loss with respect to p.
        p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
        state["t"] = t + 1  # Increment iteration number.
        return loss


weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e3)
for t in range(10):
    opt.zero_grad()  # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean()  # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward()  # Run backward pass, which computes gradients.
    opt.step()  # Run optimizer step

"""
lr=1e1
15.4683256149292
11.402588844299316
8.921308517456055
7.226259708404541
5.991397857666016
5.052948474884033
4.3178887367248535
3.72883677482605
3.2482311725616455
"""


"""
lr=1e2
24.16925621032715
4.146788120269775
0.09924197942018509
1.4909135124391293e-16
1.6617156879322643e-18
5.595580868358303e-20
3.3333264286225533e-21
2.8595427788598595e-22
3.177269824409702e-23
"""


"""
lr=1e3
8725.1025390625
1506962.375
167633472.0
13578309632.0
856946900992.0
43992855085056.0
1892760814616576.0
6.976312801912422e+16
2.2401715161488425e+18
"""
