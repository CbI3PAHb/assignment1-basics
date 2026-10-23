import pytest
import torch
from torch import nn

from cs336_basics.checkpoint.checkpoint import (
    load_checkpoint,
    load_model_weights,
    save_checkpoint,
)
from cs336_basics.optimizer.adamw import AdamW


def _make_model(device="cpu"):
    model = nn.Linear(3, 2, device=device)
    with torch.no_grad():
        model.weight.copy_(torch.arange(6, device=device).reshape(2, 3) / 10)
        model.bias.copy_(torch.tensor([0.2, -0.1], device=device))
    return model


def _take_step(model, optimizer):
    device = model.weight.device
    inputs = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.3, 2.0]], device=device)
    targets = torch.tensor([[0.4, -0.2], [0.1, 0.6]], device=device)
    optimizer.zero_grad(set_to_none=True)
    (model(inputs) - targets).square().mean().backward()
    optimizer.step()


def _assert_training_state_equal(source_model, source_optimizer, model, optimizer):
    assert (
        optimizer.state_dict()["param_groups"]
        == source_optimizer.state_dict()["param_groups"]
    )
    for source_parameter, parameter in zip(
        source_model.parameters(), model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            parameter.cpu(), source_parameter.cpu(), rtol=0, atol=0
        )
        source_state = source_optimizer.state[source_parameter]
        state = optimizer.state[parameter]
        assert set(state) == set(source_state) == {"m", "v", "t"}
        assert state["t"] == source_state["t"]
        for key in ("m", "v"):
            assert state[key].device == parameter.device
            torch.testing.assert_close(
                state[key].cpu(), source_state[key].cpu(), rtol=0, atol=0
            )


@pytest.fixture
def trained_checkpoint(tmp_path, execution_device):
    model = _make_model(execution_device)
    optimizer = AdamW(model.parameters(), lr=0.02, betas=(0.7, 0.9), weight_decay=0.1)
    completed_steps = 3
    for _ in range(completed_steps):
        _take_step(model, optimizer)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(model, optimizer, completed_steps, path)
    return path, model, optimizer, completed_steps


def _spy_on_load(monkeypatch):
    real_load = torch.load
    calls = []

    def tracking_load(*args, **kwargs):
        calls.append((args, kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracking_load)
    return calls


def test_checkpoint_round_trip_preserves_training_state(
    trained_checkpoint, execution_device
):
    path, source_model, source_optimizer, completed_steps = trained_checkpoint
    model = _make_model(execution_device)
    optimizer = AdamW(model.parameters(), lr=0.005)

    assert load_checkpoint(path, model, optimizer) == completed_steps
    _assert_training_state_equal(source_model, source_optimizer, model, optimizer)

    # Restored moments and counters must produce the same next optimizer update.
    _take_step(source_model, source_optimizer)
    _take_step(model, optimizer)
    _assert_training_state_equal(source_model, source_optimizer, model, optimizer)


def test_checkpoint_load_maps_to_cpu_and_uses_restricted_loading(
    trained_checkpoint, monkeypatch
):
    path, _, _, completed_steps = trained_checkpoint
    model = _make_model()
    optimizer = AdamW(model.parameters())
    calls = _spy_on_load(monkeypatch)

    assert load_checkpoint(path, model, optimizer) == completed_steps

    assert calls == [((path,), {"map_location": "cpu", "weights_only": True})]


def test_model_weights_load_leaves_new_optimizer_empty(trained_checkpoint, monkeypatch):
    path, source_model, _, _ = trained_checkpoint
    model = _make_model(source_model.weight.device)
    optimizer = AdamW(model.parameters(), lr=0.005, betas=(0.8, 0.95))
    original_groups = optimizer.state_dict()["param_groups"]
    calls = _spy_on_load(monkeypatch)

    load_model_weights(path, model)

    for source_parameter, parameter in zip(
        source_model.parameters(), model.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter, source_parameter, rtol=0, atol=0)
    assert optimizer.state == {}
    assert optimizer.state_dict()["param_groups"] == original_groups
    assert calls == [((path,), {"map_location": "cpu", "weights_only": True})]


@pytest.mark.cuda
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cuda_checkpoint_resumes_on_requested_device(tmp_path, device):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source_model = _make_model("cuda")
    source_optimizer = AdamW(source_model.parameters(), lr=0.02, betas=(0.7, 0.9))
    _take_step(source_model, source_optimizer)
    path = tmp_path / "cuda_checkpoint.pt"

    save_checkpoint(source_model, source_optimizer, 1, path)

    # Saving CUDA state must not require moving the live model or its moments.
    assert all(
        parameter.device.type == "cuda" for parameter in source_model.parameters()
    )
    assert all(
        state[key].device.type == "cuda"
        for state in source_optimizer.state.values()
        for key in ("m", "v")
    )
    model = _make_model(device)
    optimizer = AdamW(model.parameters())
    assert load_checkpoint(path, model, optimizer) == 1
    _assert_training_state_equal(source_model, source_optimizer, model, optimizer)
