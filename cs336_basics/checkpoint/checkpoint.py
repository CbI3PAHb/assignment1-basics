import os
import typing

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    artifact = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "iteration": iteration,
    }
    torch.save(artifact, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    artifact = torch.load(src)
    model.load_state_dict(artifact["model_state_dict"])
    optimizer.load_state_dict(artifact["optimizer_state_dict"])
    return artifact["iteration"]
