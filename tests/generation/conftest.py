"""CPU/CUDA Engine construction for generation behavior tests without artifacts."""

import importlib
from pathlib import Path

import pytest
import torch
from torch import nn

from cs336_basics.config import EngineConfig, ModelConfig, RuntimeConfig


def _engine_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device):
    engine_module = importlib.import_module("cs336_basics.generation.engine")

    def build_engine(model: nn.Module, *, vocab_size: int = 32):
        config = EngineConfig(
            model=ModelConfig(
                vocab_size=vocab_size,
                context_length=16,
                d_model=16,
                num_layers=1,
                num_heads=2,
                d_ff=32,
            ),
            checkpoint=tmp_path / "unused.pt",
            runtime=RuntimeConfig(device=torch.device(device).type),
        )

        def build_model(model_config: ModelConfig) -> nn.Module:
            return model

        def load_model_weights(src: Path, model: nn.Module) -> None:
            pass

        monkeypatch.setattr(engine_module, "build_model", build_model)
        monkeypatch.setattr(engine_module, "load_model_weights", load_model_weights)
        return engine_module.Engine(config)

    return build_engine


@pytest.fixture
def engine_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, execution_device):
    return _engine_factory(monkeypatch, tmp_path, execution_device)


@pytest.fixture
def cpu_engine_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Admission and no-forward lifecycle checks do not require CUDA."""
    return _engine_factory(monkeypatch, tmp_path, "cpu")
