import math
from typing import Callable, Optional

import torch
import torch.nn as nn


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


# ```
# B — batch_size
# S — context_length
# V — vocab_size
# L — num_layers
# D — d_model
# H — num_heads
# F - 4D — d_ff
# ```

# B = ...
# S = 1024
# V = 50257
# L = 48
# D = 1600
# H = 25
# F = 6400


# W_q = D ** 2
# W_k = D ** 2
# W_v = D ** 2
# W_o = D ** 2

# attn_weights = W_q + W_k + W_v + W_o = 4 * D ** 2

# W_up = D * F
# W_gate = D * F
# W_down = F * D

# mlp_weigths = W_up  + W_down = 2 * D * F = 2 * D * (4 * D) = 8 * D ** 2

# rmsnorm_weights = D

# transformer_block_weights = attn_weights + mlp_weigths + 2 * rmsnorm_weights = 12 * D ** 2 + 2 * D

# embedding_weights = D * V

# lm_head = D * V

# model_weights = embedding_weights + L * transformer_block_weights + rmsnorm_weights + lm_head_weigts = D * V + L * (12 * D ** 2 + 2 * D) + D + D * V = 2 * D * V + 12 * L * D ** 2 + 2 * L * D + D = D * (2 * V + 12 * L * D + 2 * L + 1) = D * (2 * V + 1 + L * (12 * D + 2))

# # activations

# embedding_inputs = B * S
# embedding_ouputs = B * S * D

# transformer_block_activations = (
#     rmsnorm_1_output       = B * S * D
#     q                      = B * S * D
#     k                      = B * S * D
#     v                      = B * S * D
#     attention_scores       = B * H * S * S
#     attention_probs        = B * H * S * S
#     weighted_values        = B * S * D
#     output_projection      = B * S * D
#     rmsnorm_2_output       = B * S * D

#     w1_output              = B * S * F
#     silu_output            = B * S * F
#     w3_output              = B * S * F
#     gate_product           = B * S * F
#     w2_output              = B * S * D
# )

# output_rmsnorm_outputs = B * S * D
# lm_head_outputs = B * S * V
# log_sum_exp_logits = B * S
# selected_logits = B * S
# loss = B * S
# loss.mean() = 1

# activations = embedding_inputs + embedding_ouputs + L * transformer_block_activations + (output_rmsnorm_outputs + lm_head_outputs + log_sum_exp_logits + selected_logits + loss)

# so, we need
# - model_weights * parameter_bits
# - activations * activation_bits
# - gradient = model_weights * parameter_bits
# - first_moment = model_weights * parameter_bits
# - second_moment = model_weights * parameter_bits

# total = 4 * activations + 16 * weights (bytes)
