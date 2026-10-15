from unittest.mock import Mock

import pytest
import torch
from attrs import evolve

from cs336_basics.training import trainer as trainer_module
from cs336_basics.config import CheckpointConfig, LoopConfig, TrainerConfig
from cs336_basics.training.trainer import Trainer


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    instance = Trainer(
        model=model,
        optimizer=optimizer,
        next_train_batch=Mock(side_effect=AssertionError("train_step is replaced in policy tests")),
        next_eval_batch=Mock(side_effect=AssertionError("train_step is replaced in policy tests")),
        lr_schedule=lambda step: 0.1,
        completed_steps=0,
        config=TrainerConfig(loop=LoopConfig(max_steps=5)),
        output_dir=tmp_path,
    )
    monkeypatch.setattr(instance, "train_step", Mock(return_value=torch.tensor(0.0)))
    return instance


@pytest.mark.parametrize(
    "enabled, interval, final, max_steps, expected_steps",
    [
        (False, None, False, 5, []),
        (True, 2, False, 5, [2, 4]),
        (True, None, True, 5, [5]),
        (True, 2, True, 5, [2, 4, 5]),
        (True, 2, True, 4, [2, 4]),
    ],
)
def test_checkpoint_policy_saves_completed_updates_once(
    trainer, monkeypatch, enabled, interval, final, max_steps, expected_steps
):
    trainer.config = evolve(
        trainer.config,
        checkpoint=CheckpointConfig(
            save_checkpoint=enabled,
            save_checkpoint_interval=interval,
            save_checkpoint_at_the_end=final,
        ),
    )
    trainer.config = evolve(trainer.config, loop=LoopConfig(max_steps=max_steps))
    save = Mock(wraps=trainer_module.save_checkpoint)
    monkeypatch.setattr(trainer_module, "save_checkpoint", save)

    trainer.train()

    assert [call.kwargs["iteration"] for call in save.call_args_list] == expected_steps
    assert trainer.completed_steps == max_steps
    assert trainer.train_step.call_count == max_steps
    assert {path.name for path in trainer.output_dir.glob("*.pt")} == {
        f"checkpoint_{step:010d}.pt" for step in expected_steps
    }
    for step in expected_steps:
        artifact = torch.load(
            trainer.output_dir / f"checkpoint_{step:010d}.pt", map_location="cpu", weights_only=True
        )
        assert artifact["iteration"] == step
        assert artifact["model_state_dict"].keys() == trainer.model.state_dict().keys()
        assert artifact["optimizer_state_dict"] == trainer.optimizer.state_dict()


@pytest.mark.parametrize("completed_steps", [3, 4])
def test_final_save_exports_resumed_state_even_without_new_updates(trainer, monkeypatch, completed_steps):
    trainer.config = evolve(trainer.config, loop=LoopConfig(max_steps=4))
    trainer.completed_steps = completed_steps
    trainer.config = evolve(trainer.config, checkpoint=CheckpointConfig(save_checkpoint=True, save_checkpoint_interval=2, save_checkpoint_at_the_end=True))
    save = Mock(wraps=trainer_module.save_checkpoint)
    monkeypatch.setattr(trainer_module, "save_checkpoint", save)

    trainer.train()

    save.assert_called_once()
    assert save.call_args.kwargs["iteration"] == 4
    assert trainer.train_step.call_count == 4 - completed_steps
    assert (trainer.output_dir / "checkpoint_0000000004.pt").is_file()


def test_failed_update_does_not_save_a_final_checkpoint(trainer, monkeypatch):
    trainer.config = evolve(trainer.config, checkpoint=CheckpointConfig(save_checkpoint=True, save_checkpoint_at_the_end=True))
    trainer.train_step.side_effect = RuntimeError("update failed")
    save = Mock()
    monkeypatch.setattr(trainer_module, "save_checkpoint", save)

    with pytest.raises(RuntimeError, match="update failed"):
        trainer.train()

    assert trainer.completed_steps == 0
    save.assert_not_called()


def test_failed_update_preserves_previous_periodic_checkpoint(trainer, monkeypatch):
    trainer.config = evolve(trainer.config, checkpoint=CheckpointConfig(save_checkpoint=True, save_checkpoint_interval=2, save_checkpoint_at_the_end=True))
    trainer.train_step.side_effect = [torch.tensor(0.0), torch.tensor(0.0), RuntimeError("update failed")]
    save = Mock(wraps=trainer_module.save_checkpoint)
    monkeypatch.setattr(trainer_module, "save_checkpoint", save)

    with pytest.raises(RuntimeError, match="update failed"):
        trainer.train()

    assert trainer.completed_steps == 2
    save.assert_called_once()
    assert save.call_args.kwargs["iteration"] == 2
    assert {path.name for path in trainer.output_dir.glob("*.pt")} == {"checkpoint_0000000002.pt"}


@pytest.mark.parametrize("interval, expected_step", [(None, 5), (1, 1)])
def test_failed_save_propagates_without_retrying_or_repeating_updates(trainer, monkeypatch, interval, expected_step):
    trainer.config = evolve(trainer.config, checkpoint=CheckpointConfig(save_checkpoint=True, save_checkpoint_interval=interval, save_checkpoint_at_the_end=True))
    save = Mock(side_effect=OSError("checkpoint write failed"))
    monkeypatch.setattr(trainer_module, "save_checkpoint", save)

    with pytest.raises(OSError, match="checkpoint write failed"):
        trainer.train()

    assert trainer.completed_steps == expected_step
    assert trainer.train_step.call_count == expected_step
    save.assert_called_once()
    assert save.call_args.kwargs["iteration"] == expected_step
