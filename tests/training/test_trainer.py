import copy

import pytest
import torch
from attrs import evolve
from torch import nn

from cs336_basics.config import (
    CheckpointConfig,
    EvaluationConfig,
    LoopConfig,
    TrainerConfig,
)
from cs336_basics.training import trainer as trainer_module
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
        torch.random.default_generator.manual_seed(42)
        return TinyLM().double()


@pytest.fixture
def batches():
    return [
        (torch.tensor([[0, 1]]), torch.tensor([[1, 2]])),
        (torch.tensor([[2, 3]]), torch.tensor([[3, 0]])),
    ]


@pytest.fixture
def tensor_model(model, execution_device):
    return model.to(execution_device)


@pytest.fixture
def tensor_batches(batches, execution_device):
    return [tuple(tensor.to(execution_device) for tensor in batch) for batch in batches]


def make_trainer(
    model,
    tmp_path,
    batches,
    *,
    accumulation=1,
    max_steps=1,
    completed_steps=0,
    schedule=None,
):
    train_batch_iterator = iter(batches)
    eval_batch_iterator = iter(batches)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return Trainer(
        model=model,
        optimizer=optimizer,
        next_train_batch=lambda: next(train_batch_iterator),
        next_eval_batch=lambda: next(eval_batch_iterator),
        lr_schedule=schedule if schedule is not None else lambda step: 0.1,
        completed_steps=completed_steps,
        config=TrainerConfig(
            loop=LoopConfig(
                max_steps=max_steps,
                gradient_accumulation_steps=accumulation,
                max_grad_norm=1e6,
            ),
        ),
        output_dir=tmp_path / "unused-output",
    )


def test_accumulation_matches_one_combined_batch(
    tensor_model, tensor_batches, tmp_path
):
    model, batches = tensor_model, tensor_batches
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    inputs = torch.cat([batch[0] for batch in batches])
    targets = torch.cat([batch[1] for batch in batches])
    logits = reference(inputs)
    expected_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten()
    )
    expected_loss.backward()
    reference_optimizer.step()

    trainer = make_trainer(model, tmp_path, batches, accumulation=2)
    actual_loss = trainer.train_step()

    assert not actual_loss.requires_grad
    torch.testing.assert_close(actual_loss, expected_loss.detach())
    for actual_parameter, expected_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter, expected_parameter)
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize("completed_steps", [0, 2])
def test_schedule_uses_completed_update_count(
    model, batches, tmp_path, completed_steps
):
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


def test_checkpoint_failure_does_not_repeat_successful_update(
    model, batches, tmp_path, monkeypatch
):
    trainer = make_trainer(model, tmp_path, batches)
    trainer.config = evolve(
        trainer.config,
        checkpoint=CheckpointConfig(save_checkpoint=True, save_checkpoint_interval=1),
    )

    def fail_save(**kwargs):
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(trainer_module, "save_checkpoint", fail_save)
    with pytest.raises(OSError, match="checkpoint write failed"):
        trainer.train()

    assert trainer.completed_steps == 1


def test_clipping_limits_accumulated_gradient_norm(
    tensor_model, tensor_batches, tmp_path
):
    model, batches = tensor_model, tensor_batches
    trainer = make_trainer(model, tmp_path, batches, accumulation=2)
    trainer.config = evolve(
        trainer.config,
        loop=LoopConfig(max_steps=1, gradient_accumulation_steps=2, max_grad_norm=0.01),
    )
    trainer.train_step()

    norms = [
        torch.linalg.vector_norm(parameter.grad) for parameter in model.parameters()
    ]
    norm = torch.linalg.vector_norm(torch.stack(norms))
    assert norm <= trainer.config.loop.max_grad_norm + 1e-8


@pytest.mark.parametrize("was_training", [False, True])
def test_evaluate_reports_token_weighted_loss_without_changing_training_state(
    tensor_model, tensor_batches, tmp_path, was_training
):
    model, batches = tensor_model, tensor_batches
    eval_batches = [batches[0], tuple(tensor.repeat(2, 1) for tensor in batches[1])]
    trainer = make_trainer(model, tmp_path, eval_batches)
    trainer.config = evolve(
        trainer.config,
        evaluation=EvaluationConfig(do_eval=True, eval_interval=2, num_batches=2),
    )
    model.train(was_training)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    original = copy.deepcopy(model.state_dict())
    original_grads = [parameter.grad.clone() for parameter in model.parameters()]
    original_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    inputs = torch.cat([batch[0] for batch in eval_batches])
    targets = torch.cat([batch[1] for batch in eval_batches])
    expected_loss = torch.nn.functional.cross_entropy(
        model(inputs).flatten(0, 1), targets.flatten()
    ).detach()
    iterator = iter(eval_batches)

    def next_eval_batch():
        assert not model.training
        assert torch.is_inference_mode_enabled()
        return next(iterator)

    trainer.next_eval_batch = next_eval_batch
    metrics = trainer.evaluate()

    torch.testing.assert_close(metrics["evaluation_loss"], expected_loss)
    torch.testing.assert_close(metrics["evaluation_perplexity"], expected_loss.exp())
    assert not metrics["evaluation_loss"].requires_grad
    assert model.training is was_training
    assert trainer.completed_steps == 0
    assert trainer.optimizer.state_dict() == original_optimizer
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
    for parameter, gradient in zip(model.parameters(), original_grads, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


@pytest.mark.parametrize("was_training", [False, True])
def test_evaluate_restores_model_mode_after_failure(model, tmp_path, was_training):
    trainer = make_trainer(model, tmp_path, [])
    trainer.config = evolve(
        trainer.config,
        evaluation=EvaluationConfig(do_eval=True, eval_interval=2, num_batches=1),
    )
    model.train(was_training)

    with pytest.raises(StopIteration):
        trainer.evaluate()

    assert model.training is was_training


@pytest.mark.parametrize("completed_steps", [0, 1])
def test_evaluation_runs_at_completed_update_intervals(
    model, batches, tmp_path, monkeypatch, completed_steps
):
    trainer = make_trainer(
        model, tmp_path, batches * 2, max_steps=4, completed_steps=completed_steps
    )
    trainer.config = evolve(
        trainer.config,
        evaluation=EvaluationConfig(do_eval=True, eval_interval=2, num_batches=1),
    )
    evaluated_steps = []

    def evaluate():
        evaluated_steps.append(trainer.completed_steps)
        loss = torch.tensor(2.0)
        return {
            "evaluation_loss": loss,
            "evaluation_perplexity": loss.exp(),
        }

    monkeypatch.setattr(trainer, "evaluate", evaluate)

    trainer.train()

    assert evaluated_steps == [2, 4]


def test_disabled_evaluation_is_not_called(model, batches, tmp_path, monkeypatch):
    trainer = make_trainer(model, tmp_path, batches)

    def unexpected_evaluation():
        raise AssertionError("Evaluation is disabled")

    monkeypatch.setattr(trainer, "evaluate", unexpected_evaluation)
    trainer.train()
    assert trainer.completed_steps == 1


def test_evaluation_failure_keeps_successful_update_count(
    model, batches, tmp_path, monkeypatch
):
    trainer = make_trainer(model, tmp_path, batches)
    trainer.config = evolve(
        trainer.config,
        evaluation=EvaluationConfig(do_eval=True, eval_interval=1, num_batches=1),
    )

    def fail_evaluation():
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(trainer, "evaluate", fail_evaluation)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        trainer.train()

    assert trainer.completed_steps == 1
