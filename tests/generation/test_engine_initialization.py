"""Constructor regression coverage independent of any serving entrypoint."""

import importlib

import torch
from torch import nn

from cs336_basics.config import EngineConfig, ModelConfig, RuntimeConfig
from cs336_basics.generation.sampling import Sampler


def test_engine_builds_and_loads_once_then_retains_eval_model_on_requested_device(
    monkeypatch,
    tmp_path,
    execution_device,
):
    module = importlib.import_module("cs336_basics.generation.engine")
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.ones((4, 4))))
    model.register_buffer("anchor", torch.ones(4))
    config = EngineConfig(
        model=ModelConfig(
            vocab_size=32,
            context_length=16,
            d_model=4,
            num_layers=1,
            num_heads=2,
            d_ff=8,
        ),
        checkpoint=tmp_path / "unused.pt",
        runtime=RuntimeConfig(device=execution_device.type),
    )
    built, loaded = [], []

    def build(config):
        built.append(config)
        return model

    def load(*, src, model):
        loaded.append((src, model))

    monkeypatch.setattr(module, "build_model", build)
    monkeypatch.setattr(module, "load_model_weights", load)
    engine = module.Engine(config)

    assert built == [config.model]
    assert loaded == [(config.checkpoint, model)]
    assert engine.model is model
    assert not model.training
    assert all(parameter.device == execution_device for parameter in model.parameters())
    assert all(buffer.device == execution_device for buffer in model.buffers())
    assert isinstance(engine.sampler, Sampler)
    assert engine.sampler.vocab_size == config.model.vocab_size
    assert torch.device(engine.sampler.device).type == execution_device.type
