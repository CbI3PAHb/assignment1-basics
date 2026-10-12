import copy
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from cs336_basics import cli
from cs336_basics.training import run as training_run
from cs336_basics.training.config.io import load_experiment


class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.projection = nn.Linear(d_model, vocab_size)
        self.first_forward_state = None

    def forward(self, inputs):
        if self.first_forward_state is None:
            self.first_forward_state = copy.deepcopy(self.state_dict())
        return self.projection(self.embedding(inputs))


@pytest.fixture
def built_components(monkeypatch):
    """Record tiny models and real optimizer state without replacing training."""
    components = {"models": [], "optimizers": []}
    real_build_optimizer = training_run.build_optimizer

    def build_model(config):
        model = TinyLM(config.vocab_size, config.d_model)
        components["models"].append(
            {"model": model, "initial_state": copy.deepcopy(model.state_dict())}
        )
        return model

    def build_optimizer(config, parameters):
        optimizer = real_build_optimizer(config, parameters)
        record = {"optimizer": optimizer, "first_step_state": None}
        components["optimizers"].append(record)
        real_step = optimizer.step

        def step(*args, **kwargs):
            if record["first_step_state"] is None:
                record["first_step_state"] = copy.deepcopy(optimizer.state_dict())
            return real_step(*args, **kwargs)

        monkeypatch.setattr(optimizer, "step", step)
        return optimizer

    monkeypatch.setattr(training_run, "build_model", build_model)
    monkeypatch.setattr(training_run, "build_optimizer", build_optimizer)
    return components


def _write_tokens(raw, count=96):
    tokens = (np.arange(count) % raw["model"]["vocab_size"]).astype(np.uint16)
    tokens.tofile(raw["data"]["train"]["path"])


@pytest.fixture
def cpu_experiment(minimal_experiment, built_components):
    minimal_experiment["runtime"] = {"device": "cpu"}
    minimal_experiment["optimizer"] = {"lr": 0.02}
    minimal_experiment["scheduler"] = {"min_lr": 0.001}
    minimal_experiment["checkpoint"] = {"save_every_steps": 2}
    minimal_experiment["loop"]["gradient_accumulation_steps"] = 2
    _write_tokens(minimal_experiment)
    return minimal_experiment


def _run_cli(raw, write_experiment):
    config_path = write_experiment(raw)
    cli.main(["train", "run", "--config", str(config_path)])
    return config_path


def _saved_checkpoints(output_dir):
    checkpoints = [
        (path, torch.load(path, map_location="cpu", weights_only=True))
        for path in output_dir.glob("*.pt")
    ]
    return sorted(checkpoints, key=lambda checkpoint: checkpoint[1]["iteration"])


def _assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_model_changed(record):
    assert any(
        not torch.equal(record["initial_state"][key], value)
        for key, value in record["model"].state_dict().items()
    )


def test_run_trains_and_saves_completed_updates_and_resolved_config(
    cpu_experiment, write_experiment, built_components
):
    config_path = _run_cli(cpu_experiment, write_experiment)
    output_dir = Path(cpu_experiment["output_dir"])
    checkpoints = _saved_checkpoints(output_dir)

    assert [artifact["iteration"] for _, artifact in checkpoints] == [2, 4]
    model_record = built_components["models"][-1]
    optimizer = built_components["optimizers"][-1]["optimizer"]
    _assert_model_changed(model_record)
    _assert_state_equal(checkpoints[-1][1]["model_state_dict"], model_record["model"].state_dict())
    _assert_state_equal(checkpoints[-1][1]["optimizer_state_dict"], optimizer.state_dict())
    assert checkpoints[-1][1]["optimizer_state_dict"]["state"]

    resolved_path = output_dir / "config.yaml"
    assert resolved_path.is_file()
    assert load_experiment(resolved_path) == load_experiment(config_path)


def test_run_with_periodic_saving_disabled_still_trains(
    cpu_experiment, write_experiment, built_components
):
    cpu_experiment["checkpoint"]["save_every_steps"] = None
    _run_cli(cpu_experiment, write_experiment)

    output_dir = Path(cpu_experiment["output_dir"])
    assert _saved_checkpoints(output_dir) == []
    assert (output_dir / "config.yaml").is_file()
    _assert_model_changed(built_components["models"][-1])


def test_missing_dataset_allows_retry_with_same_output_directory(
    cpu_experiment, write_experiment
):
    Path(cpu_experiment["data"]["train"]["path"]).unlink()
    output_dir = Path(cpu_experiment["output_dir"])

    with pytest.raises(RuntimeError):
        _run_cli(cpu_experiment, write_experiment)
    assert not output_dir.exists()

    _write_tokens(cpu_experiment)
    _run_cli(cpu_experiment, write_experiment)
    assert [artifact["iteration"] for _, artifact in _saved_checkpoints(output_dir)] == [2, 4]


@pytest.mark.parametrize("token_count", [1, 8])
def test_too_short_dataset_does_not_create_output(
    token_count, cpu_experiment, write_experiment
):
    _write_tokens(cpu_experiment, count=token_count)

    with pytest.raises(ValueError):
        _run_cli(cpu_experiment, write_experiment)

    assert not Path(cpu_experiment["output_dir"]).exists()


def test_unavailable_cuda_does_not_create_output(
    cpu_experiment, write_experiment, monkeypatch
):
    cpu_experiment["runtime"]["device"] = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        _run_cli(cpu_experiment, write_experiment)

    assert not Path(cpu_experiment["output_dir"]).exists()


@pytest.mark.parametrize("initialization", ["resume_from", "init_weights_from"])
@pytest.mark.parametrize("artifact_kind", ["missing", "corrupt"])
def test_invalid_checkpoint_does_not_create_output(
    initialization, artifact_kind, cpu_experiment, write_experiment, tmp_path
):
    checkpoint_path = tmp_path / "invalid.pt"
    if artifact_kind == "corrupt":
        checkpoint_path.write_bytes(b"not a torch checkpoint")
        expected_errors = (pickle.UnpicklingError, RuntimeError, EOFError)
    else:
        expected_errors = (FileNotFoundError,)
    cpu_experiment["checkpoint"][initialization] = str(checkpoint_path)

    with pytest.raises(expected_errors):
        _run_cli(cpu_experiment, write_experiment)

    assert not Path(cpu_experiment["output_dir"]).exists()


def test_existing_output_directory_is_preserved(cpu_experiment, write_experiment):
    output_dir = Path(cpu_experiment["output_dir"])
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("existing run", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _run_cli(cpu_experiment, write_experiment)

    assert sentinel.read_text(encoding="utf-8") == "existing run"
    assert list(output_dir.iterdir()) == [sentinel]


def test_output_creation_race_preserves_the_other_run(
    cpu_experiment, write_experiment, monkeypatch
):
    output_dir = Path(cpu_experiment["output_dir"])
    sentinel = output_dir / "other-run.txt"
    real_mkdir = Path.mkdir
    claimed = False

    def competing_mkdir(path, mode=0o777, parents=False, exist_ok=False):
        nonlocal claimed
        if path == output_dir and not claimed:
            claimed = True
            real_mkdir(output_dir, parents=True, exist_ok=False)
            sentinel.write_text("other invocation", encoding="utf-8")
        return real_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", competing_mkdir)

    with pytest.raises(FileExistsError):
        _run_cli(cpu_experiment, write_experiment)

    assert claimed
    assert sentinel.read_text(encoding="utf-8") == "other invocation"
    assert list(output_dir.iterdir()) == [sentinel]


def test_config_open_failure_removes_new_output_directory(
    cpu_experiment, write_experiment, monkeypatch
):
    output_dir = Path(cpu_experiment["output_dir"])
    config_path = output_dir / "config.yaml"
    real_open = Path.open
    attempted_modes = []

    def fail_config_open(path, mode="r", *args, **kwargs):
        if path == config_path:
            attempted_modes.append(mode)
            raise OSError("cannot open run config")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_config_open)

    with pytest.raises(OSError, match="cannot open run config"):
        _run_cli(cpu_experiment, write_experiment)

    assert attempted_modes == ["x"]
    assert not output_dir.exists()


def test_config_write_failure_removes_owned_file_and_new_output_directory(
    cpu_experiment, write_experiment, monkeypatch
):
    output_dir = Path(cpu_experiment["output_dir"])
    config_path = output_dir / "config.yaml"
    real_open = Path.open
    writes = []

    class FailingWriter:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def write(self, text):
            self.file.write(text[:10])
            writes.append(config_path)
            raise OSError("cannot finish run config")

    def fail_config_write(path, *args, **kwargs):
        file = real_open(path, *args, **kwargs)
        return FailingWriter(file) if path == config_path else file

    monkeypatch.setattr(Path, "open", fail_config_write)

    with pytest.raises(OSError, match="cannot finish run config"):
        _run_cli(cpu_experiment, write_experiment)

    assert writes == [config_path]
    assert not output_dir.exists()


def test_training_failure_preserves_output_directory_and_resolved_config(
    cpu_experiment, write_experiment, monkeypatch
):
    def fail_training(self):
        raise RuntimeError("training interrupted")

    monkeypatch.setattr(training_run.Trainer, "train", fail_training)

    with pytest.raises(RuntimeError, match="training interrupted"):
        _run_cli(cpu_experiment, write_experiment)

    output_dir = Path(cpu_experiment["output_dir"])
    assert output_dir.is_dir()
    resolved_config = load_experiment(output_dir / "config.yaml")
    assert resolved_config.output_dir == output_dir
    assert resolved_config.runtime.device == "cpu"
    assert resolved_config.loop.max_steps == cpu_experiment["loop"]["max_steps"]
    assert _saved_checkpoints(output_dir) == []


def test_resume_restores_optimizer_and_saves_at_absolute_update_count(
    cpu_experiment, write_experiment, built_components, tmp_path
):
    cpu_experiment["loop"]["max_steps"] = 2
    cpu_experiment["output_dir"] = str(tmp_path / "source")
    _run_cli(cpu_experiment, write_experiment)
    source_path, source_artifact = _saved_checkpoints(Path(cpu_experiment["output_dir"]))[0]

    cpu_experiment["loop"]["max_steps"] = 4
    cpu_experiment["output_dir"] = str(tmp_path / "resumed")
    cpu_experiment["checkpoint"]["resume_from"] = str(source_path)
    _run_cli(cpu_experiment, write_experiment)

    checkpoints = _saved_checkpoints(Path(cpu_experiment["output_dir"]))
    assert [artifact["iteration"] for _, artifact in checkpoints] == [4]
    optimizer_record = built_components["optimizers"][-1]
    _assert_state_equal(
        optimizer_record["first_step_state"]["state"],
        source_artifact["optimizer_state_dict"]["state"],
    )
    _assert_state_equal(
        checkpoints[0][1]["optimizer_state_dict"], optimizer_record["optimizer"].state_dict()
    )
    _assert_state_equal(
        built_components["models"][-1]["model"].first_forward_state,
        source_artifact["model_state_dict"],
    )


def test_init_weights_restores_model_but_resets_optimizer_and_update_count(
    cpu_experiment, write_experiment, built_components, tmp_path
):
    cpu_experiment["loop"]["max_steps"] = 2
    cpu_experiment["output_dir"] = str(tmp_path / "source")
    _run_cli(cpu_experiment, write_experiment)
    source_path, source_artifact = _saved_checkpoints(Path(cpu_experiment["output_dir"]))[0]

    cpu_experiment["loop"]["max_steps"] = 1
    cpu_experiment["output_dir"] = str(tmp_path / "initialized")
    cpu_experiment["checkpoint"] = {
        "init_weights_from": str(source_path),
        "save_every_steps": 1,
    }
    _run_cli(cpu_experiment, write_experiment)

    checkpoints = _saved_checkpoints(Path(cpu_experiment["output_dir"]))
    assert [artifact["iteration"] for _, artifact in checkpoints] == [1]
    optimizer_record = built_components["optimizers"][-1]
    assert optimizer_record["first_step_state"]["state"] == {}
    assert checkpoints[0][1]["optimizer_state_dict"]["state"]
    _assert_state_equal(
        built_components["models"][-1]["model"].first_forward_state,
        source_artifact["model_state_dict"],
    )
