import pytest
import yaml


@pytest.fixture
def minimal_experiment(tmp_path):
    """A complete minimal YAML config whose resource paths need not exist"""
    return {
        "output_dir": str(tmp_path / "run"),
        "model": {
            "vocab_size": 32,
            "context_length": 8,
            "d_model": 8,
            "num_layers": 1,
            "num_heads": 2,
            "d_ff": 16,
        },
        "data": {
            "train": {"path": str(tmp_path / "missing-train-tokens.bin")},
            "eval": {"path": str(tmp_path / "missing-eval-tokens.bin")},
            "batching": {"sequence_length": 8, "micro_batch_size": 2},
        },
        "optimizer": {},
        "loop": {"max_steps": 4},
    }


@pytest.fixture
def write_experiment(tmp_path):
    def write(raw):
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return path

    return write


@pytest.fixture
def experiment_path(minimal_experiment, write_experiment):
    return write_experiment(minimal_experiment)
