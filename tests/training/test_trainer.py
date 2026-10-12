import copy

import pytest
import torch
from torch import nn

from cs336_basics.training import trainer as trainer_module
from cs336_basics.training.config import CheckpointConfig, LoopConfig
from cs336_basics.training.trainer import Trainer


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 3)
        self.output = nn.Linear(3, 4)

    def forward(self, inputs):
        return self.output(self.embedding(inputs))


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return TinyLM().double()


@pytest.fixture
def batches():
    return [
        (torch.tensor([[0, 1]]), torch.tensor([[1, 2]])),
        (torch.tensor([[2, 3]]), torch.tensor([[3, 0]])),
    ]


def make_trainer(model, tmp_path, batches, *, accumulation=1, max_steps=1, completed_steps=0, schedule=None):
    batch_iterator = iter(batches)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return Trainer(
        loop=LoopConfig(max_steps=max_steps, gradient_accumulation_steps=accumulation, max_grad_norm=1e6),
        model=model,
        optimizer=optimizer,
        next_batch=lambda: next(batch_iterator),
        lr_schedule=schedule if schedule is not None else lambda step: 0.1,
        completed_steps=completed_steps,
        checkpoint=CheckpointConfig(),
        output_dir=tmp_path / "unused-output",
    )


def test_accumulation_matches_one_combined_batch(model, batches, tmp_path):
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    inputs = torch.cat([batch[0] for batch in batches])
    targets = torch.cat([batch[1] for batch in batches])
    logits = reference(inputs)
    expected_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    expected_loss.backward()
    reference_optimizer.step()

    trainer = make_trainer(model, tmp_path, batches, accumulation=2)
    actual_loss = trainer.train_step()

    assert not actual_loss.requires_grad
    torch.testing.assert_close(actual_loss, expected_loss.detach())
    for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter, expected_parameter)
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize("completed_steps", [0, 2])
def test_schedule_uses_completed_update_count(model, batches, tmp_path, completed_steps):
    seen_steps = []

    def schedule(step):
        seen_steps.append(step)
        return 0.1

    trainer = make_trainer(
        model,
        tmp_path,
        batches * 2,
        max_steps=4,
        completed_steps=completed_steps,
        schedule=schedule,
    )
    trainer.train()

    assert seen_steps == list(range(completed_steps, 4))
    assert trainer.completed_steps == 4
    assert not trainer.output_dir.exists()


def test_failed_update_does_not_advance_counter(model, batches, tmp_path):
    trainer = make_trainer(model, tmp_path, [])
    original = {name: value.clone() for name, value in model.state_dict().items()}

    with pytest.raises(StopIteration):
        trainer.train()

    assert trainer.completed_steps == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_checkpoint_failure_does_not_repeat_successful_update(model, batches, tmp_path, monkeypatch):
    trainer = make_trainer(model, tmp_path, batches)
    trainer.checkpoint = CheckpointConfig(save_every_steps=1)

    def fail_save(**kwargs):
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(trainer_module, "save_checkpoint", fail_save)
    with pytest.raises(OSError, match="checkpoint write failed"):
        trainer.train()

    assert trainer.completed_steps == 1


def test_clipping_limits_accumulated_gradient_norm(model, batches, tmp_path):
    trainer = make_trainer(model, tmp_path, batches, accumulation=2)
    trainer.loop = LoopConfig(max_steps=1, gradient_accumulation_steps=2, max_grad_norm=0.01)
    trainer.train_step()

    norms = [torch.linalg.vector_norm(parameter.grad) for parameter in model.parameters()]
    norm = torch.linalg.vector_norm(torch.stack(norms))
    assert norm <= trainer.loop.max_grad_norm + 1e-8
