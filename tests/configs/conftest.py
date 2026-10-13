import pytest

from cs336_basics.training.config._cattrs import _make_converter


@pytest.fixture
def converter():
    return _make_converter()


@pytest.fixture
def raw_experiment(tmp_path):
    """A fresh minimal experiment; its paths deliberately do not exist."""
    return {
        "output_dir": str(tmp_path / "run"),
        "model": {
            "vocab_size": 1024,
            "context_length": 128,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "d_ff": 128,
        },
        "data": {
            "train": {"path": str(tmp_path / "train_tokens.bin")},
            "eval": {"path": str(tmp_path / "eval_tokens.bin")},
            "batching": {"sequence_length": 64, "micro_batch_size": 2},
        },
        "optimizer": {},
        "loop": {"max_steps": 100},
        "logging": {"silent": False},
    }
