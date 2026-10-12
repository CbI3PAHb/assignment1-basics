import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[
    Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]
]:
    indexes = torch.randint(
        low=0,
        high=dataset.shape[0] - context_length,
        size=(batch_size,),
    )
    # Copy only the sampled windows; the source may be a read-only memory map.
    windows = np.stack(
        [dataset[i : i + context_length + 1] for i in indexes.tolist()],
        dtype=np.int64,
    )
    inputs = torch.from_numpy(windows[:, :-1].copy())
    targets = torch.from_numpy(windows[:, 1:].copy())

    if "cuda" in device:
        inputs = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs = inputs.to(device)
        targets = targets.to(device)

    return inputs, targets
