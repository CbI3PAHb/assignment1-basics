import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    indexes = torch.randint(
        low=0,
        high=dataset.shape[0] - context_length,
        size=(batch_size,),
    )
    inputs: Int[Tensor, "batch_size context_length"] = torch.stack(
        [torch.from_numpy(dataset[i : i + context_length]).to(torch.int64) for i in indexes]
    )

    targets: Int[Tensor, "batch_size context_length"] = torch.stack(
        [torch.from_numpy(dataset[i + 1 : i + 1 + context_length]).to(torch.int64) for i in indexes]
    )
    if "cuda" in device:
        inputs = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs = inputs.to(device)
        targets = targets.to(device)

    return inputs, targets
