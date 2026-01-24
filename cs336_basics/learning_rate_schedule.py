import math

"""
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """


def learning_rate_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        # Linear warm-up phase
        lr = it / warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        # Constant learning rate phase
        lr = min_learning_rate + 1 / 2 * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        ) * (max_learning_rate - min_learning_rate)
    else:
        # Cosine decay phase
        lr = min_learning_rate
    return lr
