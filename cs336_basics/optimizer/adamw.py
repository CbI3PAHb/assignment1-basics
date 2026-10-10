import math

import torch
import torch.nn as nn

from typing import Callable, Optional


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params=params, defaults=defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 1

                m = state["m"]
                v = state["v"]
                t = state["t"]

                grad = p.grad.data
                beta1, beta2 = group["betas"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                adjusted_lr = group["lr"] * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-adjusted_lr)
                state["t"] = t + 1  #  Increment iteration number.
        return loss
