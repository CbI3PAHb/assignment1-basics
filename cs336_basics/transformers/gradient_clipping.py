from typing import Iterable

import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return

    norms = [torch.linalg.vector_norm(g) for g in grads]
    norm = torch.linalg.vector_norm(torch.stack(norms))

    if norm > max_l2_norm:
        multiplier = max_l2_norm / (norm + eps)

        for grad in grads:
            grad.mul_(multiplier)
