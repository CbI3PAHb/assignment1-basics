import math


def cosine_lr_scheduler(
    *,
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        angle = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
        return min_learning_rate + 0.5 * (1 + math.cos(angle)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
