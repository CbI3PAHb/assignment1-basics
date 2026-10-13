import pytest

from cs336_basics.training.config import (
    CheckpointConfig,
    EvaluationConfig,
    LoggingConfig,
    LoopConfig,
    TrainerConfig,
)


def test_trainer_config_needs_no_model_or_dataset_settings():
    config = TrainerConfig(loop=LoopConfig(max_steps=1))

    assert config.loop.max_steps == 1
    assert config.checkpoint == CheckpointConfig()
    assert config.evaluation == EvaluationConfig()
    assert config.logging == LoggingConfig()


@pytest.mark.parametrize("field", ["loop", "checkpoint", "evaluation", "logging"])
def test_trainer_config_rejects_unstructured_nested_settings(field):
    arguments = {"loop": LoopConfig(max_steps=1), field: {}}

    with pytest.raises(TypeError, match=field):
        TrainerConfig(**arguments)
