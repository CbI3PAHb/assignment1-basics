from pathlib import Path

import pytest
from attrs import evolve
from attrs.exceptions import FrozenInstanceError

from cs336_basics.training.config import (
    BatchingConfig,
    CheckpointConfig,
    ConfigError,
    DataConfig,
    DatasetConfig,
    ExperimentConfig,
    LoopConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    SchedulerConfig,
)


@pytest.fixture
def experiment(tmp_path):
    return ExperimentConfig(
        output_dir=str(tmp_path / "run"),
        model=ModelConfig(
            vocab_size=1024,
            context_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        ),
        data=DataConfig(
            train=DatasetConfig(path="tokens.bin"),
            batching=BatchingConfig(sequence_length=64, micro_batch_size=2),
        ),
        optimizer=OptimizerConfig(betas=[0.9, 0.95]),
        loop=LoopConfig(max_steps=100),
    )


def test_configuration_converts_serializable_inputs(experiment):
    assert isinstance(experiment.output_dir, Path)
    assert experiment.data.train.path == Path("tokens.bin")
    assert experiment.optimizer.betas == (0.9, 0.95)
    assert isinstance(experiment.optimizer.betas, tuple)
    assert CheckpointConfig(resume_from="run/checkpoint.pt").resume_from == Path(
        "run/checkpoint.pt"
    )
    assert CheckpointConfig().resume_from is None


@pytest.mark.parametrize(
    "lr", [True, "0.001", 0, -1, float("nan"), float("inf"), float("-inf")]
)
def test_optimizer_rejects_invalid_learning_rate(lr):
    with pytest.raises((TypeError, ValueError)):
        OptimizerConfig(lr=lr)


def test_numeric_configuration_accepts_large_finite_integer():
    value = 10**400
    assert OptimizerConfig(lr=value).lr == value


@pytest.mark.parametrize("steps", [True, 1.5, "10", 0, -1])
def test_loop_requires_positive_integer_updates(steps):
    with pytest.raises((TypeError, ValueError)):
        LoopConfig(max_steps=steps)


def test_model_checks_attention_dimensions(experiment):
    with pytest.raises(ConfigError, match="divisible"):
        evolve(experiment.model, d_model=66)
    with pytest.raises(ConfigError, match="even"):
        evolve(experiment.model, d_model=12)


@pytest.mark.parametrize(
    "betas",
    [[0.9], [0.9, 0.95, 0.99], [True, 0.9], [-0.1, 0.9], [0.9, 1], [0.9, float("nan")]],
)
def test_optimizer_rejects_invalid_betas(betas):
    with pytest.raises((TypeError, ValueError)):
        OptimizerConfig(betas=betas)


def test_zero_is_valid_for_nonnegative_settings():
    assert SchedulerConfig(min_lr=0, warmup_steps=0).min_lr == 0
    assert OptimizerConfig(weight_decay=0, betas=[0, 0]).weight_decay == 0
    with pytest.raises(ValueError):
        SchedulerConfig(min_lr=-1)
    with pytest.raises(ValueError):
        CheckpointConfig(save_every_steps=0)


def test_experiment_checks_component_compatibility(experiment):
    oversized_batching = evolve(experiment.data.batching, sequence_length=129)
    with pytest.raises(ConfigError, match="sequence_length"):
        evolve(experiment, data=evolve(experiment.data, batching=oversized_batching))
    with pytest.raises(ConfigError, match="min_lr"):
        evolve(
            experiment, scheduler=SchedulerConfig(min_lr=experiment.optimizer.lr * 2)
        )
    with pytest.raises(ConfigError, match="warmup_steps"):
        evolve(
            experiment,
            scheduler=SchedulerConfig(warmup_steps=experiment.loop.max_steps),
        )

    compatible = evolve(
        experiment,
        data=evolve(
            experiment.data,
            batching=evolve(experiment.data.batching, sequence_length=128),
        ),
        scheduler=SchedulerConfig(min_lr=experiment.optimizer.lr, warmup_steps=99),
    )
    assert compatible.data.batching.sequence_length == compatible.model.context_length


def test_checkpoint_start_modes_are_mutually_exclusive():
    with pytest.raises(ConfigError, match="mutually exclusive"):
        CheckpointConfig(resume_from="resume.pt", init_weights_from="weights.pt")
    assert CheckpointConfig(init_weights_from="weights.pt").init_weights_from == Path(
        "weights.pt"
    )


def test_configuration_is_frozen_and_evolve_revalidates(experiment):
    with pytest.raises(FrozenInstanceError):
        experiment.seed = 0
    with pytest.raises(ValueError):
        evolve(experiment.loop, max_steps=0)
    assert evolve(experiment.loop, max_steps=200).max_steps == 200
    assert experiment.loop.max_steps == 100


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_experiment_rejects_seed_outside_uint32(experiment, seed):
    with pytest.raises(ValueError):
        evolve(experiment, seed=seed)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_experiment_accepts_seed_boundaries(experiment, seed):
    assert evolve(experiment, seed=seed).seed == seed


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runtime_accepts_supported_device_types(device):
    assert RuntimeConfig(device=device).device == device


@pytest.mark.parametrize("device", ["cpu:0", "cuda:0", "cuda:1", "mps", "", None])
def test_runtime_rejects_indices_and_unsupported_devices(device):
    with pytest.raises(ValueError):
        RuntimeConfig(device=device)
