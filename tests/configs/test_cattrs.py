"""Behavioral coverage for strict conversion of serialized configurations."""

from copy import deepcopy
from pathlib import Path

import cattrs
import pytest
from cattrs.errors import ClassValidationError, ForbiddenExtraKeysError

from cs336_basics.config import ConfigError, ExperimentConfig
from cs336_basics.config.io import _make_converter

from ._helpers import exception_leaves


def _mapping_at(raw, path):
    mapping = raw
    for field in path:
        mapping = mapping.setdefault(field, {})
    return mapping


@pytest.mark.parametrize("value", [128.10, 128.0, "128", True, False, None])
def test_integer_conversion_rejects_coercible_nonintegers(converter, value):
    with pytest.raises(TypeError, match="expected an integer"):
        converter.structure(value, int)


@pytest.mark.parametrize("value", [0, 1, 128])
def test_integer_conversion_preserves_integers(converter, value):
    actual = converter.structure(value, int)
    assert actual == value
    assert type(actual) is int


@pytest.mark.parametrize("value", [0, 1, 10_000, 10**400, 0.0, 0.9, 1e-3])
def test_float_conversion_accepts_numbers_without_coercion(converter, value):
    actual = converter.structure(value, float)
    assert actual == value
    assert type(actual) is type(value)


@pytest.mark.parametrize("value", ["0.9", True, False, None, [], {}])
def test_float_conversion_rejects_nonnumbers(converter, value):
    with pytest.raises(TypeError, match="expected int or float"):
        converter.structure(value, float)


@pytest.mark.parametrize("value", [False, True])
def test_boolean_conversion_preserves_booleans(converter, value):
    assert converter.structure(value, bool) is value


@pytest.mark.parametrize("field", ["save_checkpoint", "save_checkpoint_at_the_end"])
@pytest.mark.parametrize("value", [0, 1, 0.0, "true", "false", None, []])
def test_checkpoint_flags_reject_boolean_coercion(converter, raw_experiment, field, value):
    raw_experiment["checkpoint"] = {"save_checkpoint": True, "save_checkpoint_at_the_end": True, field: value}

    with pytest.raises(ClassValidationError) as caught:
        converter.structure(raw_experiment, ExperimentConfig)

    assert any(
        isinstance(error, TypeError)
        and any(getattr(note, "name", None) == field for note in getattr(error, "__notes__", ()))
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("data", "batching", "sequence_length"), 128.10),
        (("data", "batching", "sequence_length"), 128.0),
        (("data", "batching", "sequence_length"), "64"),
        (("data", "batching", "micro_batch_size"), True),
        (("model", "num_layers"), "2"),
        (("loop", "max_steps"), 100.0),
        (("lr_scheduler", "warmup_steps"), False),
        (("checkpoint", "save_checkpoint_interval"), 5.0),
        (("checkpoint", "save_checkpoint_interval"), "5"),
        (("checkpoint", "save_checkpoint_interval"), True),
        (("seed",), True),
    ],
)
def test_nested_integer_fields_use_strict_conversion(converter, raw_experiment, path, value):
    _mapping_at(raw_experiment, path[:-1])[path[-1]] = value

    with pytest.raises(ClassValidationError) as caught:
        converter.structure(raw_experiment, ExperimentConfig)

    assert any(
        isinstance(error, TypeError) and "expected an integer" in str(error)
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize("save_checkpoint_interval", [None, 5])
def test_optional_integer_accepts_none_or_integer(converter, raw_experiment, save_checkpoint_interval):
    raw_experiment["checkpoint"] = {
        "save_checkpoint": save_checkpoint_interval is not None,
        "save_checkpoint_interval": save_checkpoint_interval,
    }
    config = converter.structure(raw_experiment, ExperimentConfig)
    assert config.checkpoint.save_checkpoint_interval == save_checkpoint_interval


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "rope_theta"), "10000"),
        (("optimizer", "lr"), True),
        (("optimizer", "eps"), "1e-8"),
        (("optimizer", "weight_decay"), False),
        (("loop", "max_grad_norm"), "1.0"),
        (("lr_scheduler", "min_lr"), True),
    ],
)
def test_nested_float_fields_reject_coercible_nonnumbers(converter, raw_experiment, path, value):
    _mapping_at(raw_experiment, path[:-1])[path[-1]] = value

    with pytest.raises(ClassValidationError) as caught:
        converter.structure(raw_experiment, ExperimentConfig)

    assert any(
        isinstance(error, TypeError) and "expected int or float" in str(error)
        for error in exception_leaves(caught.value)
    )


def test_nested_float_fields_accept_integer_values(converter, raw_experiment):
    raw_experiment["model"]["rope_theta"] = 10_000
    raw_experiment["optimizer"] = {"lr": 1, "eps": 1, "weight_decay": 0, "betas": [0, 0.9]}
    raw_experiment["lr_scheduler"] = {"min_lr": 0}
    raw_experiment["loop"]["max_grad_norm"] = 1

    config = converter.structure(raw_experiment, ExperimentConfig)

    assert config.model.rope_theta == 10_000
    assert config.optimizer.lr == 1
    assert config.optimizer.eps == 1
    assert config.optimizer.weight_decay == 0
    assert config.optimizer.betas == (0, 0.9)
    assert config.lr_scheduler.min_lr == 0
    assert config.loop.max_grad_norm == 1


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("model",),
        ("data",),
        ("data", "train"),
        ("data", "batching"),
        ("optimizer",),
        ("loop",),
        ("lr_scheduler",),
        ("runtime",),
        ("checkpoint",),
    ],
)
def test_unknown_fields_are_rejected_at_every_config_level(converter, raw_experiment, path):
    _mapping_at(raw_experiment, path)["unexpected_field"] = 123

    with pytest.raises(ClassValidationError) as caught:
        converter.structure(raw_experiment, ExperimentConfig)

    assert any(
        isinstance(error, ForbiddenExtraKeysError) and error.extra_fields == {"unexpected_field"}
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize("path", [("model",), ("model", "d_model"), ("data", "train"), ("loop", "max_steps")])
def test_missing_required_fields_are_rejected(converter, raw_experiment, path):
    del _mapping_at(raw_experiment, path[:-1])[path[-1]]

    with pytest.raises(ClassValidationError) as caught:
        converter.structure(raw_experiment, ExperimentConfig)

    assert any(
        isinstance(error, KeyError) and error.args == (path[-1],)
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize("path", [("output_dir",), ("data", "train", "path"), ("checkpoint", "resume_from")])
@pytest.mark.parametrize("value", ["", 123, True])
def test_path_converters_see_and_reject_invalid_original_values(converter, raw_experiment, path, value):
    _mapping_at(raw_experiment, path[:-1])[path[-1]] = value

    with pytest.raises(ClassValidationError) as caught:
        converter.structure(raw_experiment, ExperimentConfig)

    assert any(
        isinstance(error, ConfigError) and "nonempty string" in str(error)
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize("value", [[0.9, 0.999], (0.9, 0.999)])
def test_beta_converter_accepts_lists_and_tuples(converter, raw_experiment, value):
    raw_experiment["optimizer"]["betas"] = value
    config = converter.structure(raw_experiment, ExperimentConfig)
    assert config.optimizer.betas == (0.9, 0.999)
    assert type(config.optimizer.betas) is tuple


@pytest.mark.parametrize("value", [{0: None, 0.9: None}, "01", None])
def test_beta_converter_rejects_other_iterables_before_tuple_structuring(converter, raw_experiment, value):
    raw_experiment["optimizer"]["betas"] = value

    with pytest.raises(ClassValidationError) as caught:
        converter.structure(raw_experiment, ExperimentConfig)

    assert any(
        isinstance(error, ConfigError) and "tuple or list" in str(error)
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize("value", [[False, 0.9], ["0.9", 0.999]])
def test_beta_members_are_validated_without_scalar_coercion(converter, raw_experiment, value):
    raw_experiment["optimizer"]["betas"] = value
    with pytest.raises(ClassValidationError) as caught:
        converter.structure(raw_experiment, ExperimentConfig)
    assert any(isinstance(error, (TypeError, ValueError)) for error in exception_leaves(caught.value))


@pytest.mark.parametrize("invalid", [False, True])
def test_structuring_does_not_mutate_input(converter, raw_experiment, invalid):
    raw_experiment["optimizer"]["betas"] = [0.9, 0.999]
    raw_experiment["checkpoint"] = {"resume_from": "weights.pt"}
    if invalid:
        raw_experiment["data"]["batching"]["unexpected_field"] = 1
    original = deepcopy(raw_experiment)

    if invalid:
        with pytest.raises(ClassValidationError):
            converter.structure(raw_experiment, ExperimentConfig)
    else:
        config = converter.structure(raw_experiment, ExperimentConfig)
        assert config.output_dir == Path(original["output_dir"])
        assert config.data.train.path == Path(original["data"]["train"]["path"])
        assert config.checkpoint.resume_from == Path("weights.pt")

    assert raw_experiment == original


def test_converter_factories_are_independent():
    customized = _make_converter()
    untouched = _make_converter()
    customized.register_structure_hook(int, lambda value, _: 7)

    assert customized.structure("anything", int) == 7
    with pytest.raises(TypeError):
        untouched.structure("7", int)


def test_local_converter_does_not_change_global_cattrs_defaults(converter, raw_experiment):
    with pytest.raises(TypeError):
        converter.structure("128", int)
    assert cattrs.structure("128", int) == 128

    with pytest.raises(TypeError):
        converter.structure("0.9", float)
    assert cattrs.structure("0.9", float) == 0.9

    with pytest.raises(TypeError):
        converter.structure("false", bool)
    assert cattrs.structure("false", bool) is True

    raw_experiment["unexpected_field"] = 123
    with pytest.raises(ClassValidationError):
        converter.structure(raw_experiment, ExperimentConfig)
    assert isinstance(cattrs.structure(raw_experiment, ExperimentConfig), ExperimentConfig)
