"""Evaluation configuration validation through Python and experiment YAML."""

import pytest
import yaml
from attrs import evolve
from attrs.exceptions import FrozenInstanceError
from cattrs.errors import ClassValidationError, ForbiddenExtraKeysError

from cs336_basics.config.evaluation import EvaluationConfig, EvaluationConfigError
from cs336_basics.config.io import dump_experiment, load_experiment

from ._helpers import exception_leaves


@pytest.fixture(params=["python", "yaml"])
def build_evaluation(request, tmp_path, raw_experiment):
    """Check the same rules for direct construction and the public YAML loader."""
    def build(options):
        if request.param == "python":
            return EvaluationConfig(**options)
        raw_experiment["evaluation"] = options
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(raw_experiment), encoding="utf-8")
        return load_experiment(path).evaluation

    return build


@pytest.mark.parametrize("options", [
    {},
    {"do_eval": False},
    {"do_eval": False, "eval_interval": None, "num_batches": None},
])
def test_disabled_evaluation_accepts_empty_settings(build_evaluation, options):
    config = build_evaluation(options)

    assert config.do_eval is False
    assert config.eval_interval is None
    assert config.num_batches is None


@pytest.mark.parametrize("eval_interval, num_batches", [(1, 1), (10, 20), (1000, 100)])
def test_enabled_evaluation_preserves_positive_settings(build_evaluation, eval_interval, num_batches):
    config = build_evaluation({"do_eval": True, "eval_interval": eval_interval, "num_batches": num_batches})

    assert config.do_eval is True
    assert config.eval_interval == eval_interval
    assert config.num_batches == num_batches
    assert type(config.eval_interval) is int
    assert type(config.num_batches) is int


@pytest.mark.parametrize("field", ["eval_interval", "num_batches"])
@pytest.mark.parametrize("explicit_null", [False, True], ids=["omitted", "null"])
def test_enabled_evaluation_requires_both_settings(build_evaluation, field, explicit_null):
    options = {"do_eval": True, "eval_interval": 10, "num_batches": 20}
    if explicit_null:
        options[field] = None
    else:
        del options[field]

    with pytest.raises((EvaluationConfigError, ClassValidationError)) as caught:
        build_evaluation(options)

    assert any(
        isinstance(error, EvaluationConfigError) and field in str(error)
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize("settings", [
    {"eval_interval": 10},
    {"num_batches": 20},
    {"eval_interval": 10, "num_batches": 20},
])
def test_disabled_evaluation_rejects_active_settings(build_evaluation, settings):
    with pytest.raises((EvaluationConfigError, ClassValidationError)) as caught:
        build_evaluation({"do_eval": False, **settings})

    assert any(isinstance(error, EvaluationConfigError) for error in exception_leaves(caught.value))


@pytest.mark.parametrize("field", ["eval_interval", "num_batches"])
@pytest.mark.parametrize("value", [0, -1, 1.0, 1.5, "2", True, False])
def test_evaluation_requires_positive_integers_without_coercion(build_evaluation, field, value):
    options = {"do_eval": True, "eval_interval": 10, "num_batches": 20}
    options[field] = value

    with pytest.raises((TypeError, ValueError, ClassValidationError)) as caught:
        build_evaluation(options)

    assert any(isinstance(error, (TypeError, ValueError)) for error in exception_leaves(caught.value))


@pytest.mark.parametrize("value", [0, 1, "true", "false", None])
def test_do_eval_requires_a_boolean_without_coercion(build_evaluation, value):
    with pytest.raises((TypeError, ClassValidationError)) as caught:
        build_evaluation({"do_eval": value, "eval_interval": 10, "num_batches": 20})

    assert any(isinstance(error, TypeError) for error in exception_leaves(caught.value))


def test_evaluation_is_frozen_and_evolve_revalidates():
    config = EvaluationConfig(do_eval=True, eval_interval=10, num_batches=20)

    with pytest.raises(FrozenInstanceError):
        config.num_batches = 0
    with pytest.raises(EvaluationConfigError):
        evolve(config, do_eval=False)

    disabled = evolve(config, do_eval=False, eval_interval=None, num_batches=None)
    assert disabled == EvaluationConfig()
    assert config.do_eval is True


def test_omitted_evaluation_section_defaults_to_disabled(tmp_path, raw_experiment):
    raw_experiment.pop("evaluation", None)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw_experiment), encoding="utf-8")

    assert load_experiment(path).evaluation == EvaluationConfig()


def test_enabled_evaluation_survives_yaml_round_trip_without_creating_artifacts(tmp_path, raw_experiment):
    settings = {"do_eval": True, "eval_interval": 10, "num_batches": 20}
    raw_experiment["evaluation"] = settings
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw_experiment), encoding="utf-8")
    config = load_experiment(path)

    dumped = dump_experiment(config)
    assert yaml.safe_load(dumped)["evaluation"] == settings
    path.write_text(dumped, encoding="utf-8")
    assert load_experiment(path) == config
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("old_name", ["interval", "steps"])
def test_yaml_rejects_old_evaluation_field_names(tmp_path, raw_experiment, old_name):
    raw_experiment["evaluation"] = {old_name: 10}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw_experiment), encoding="utf-8")

    with pytest.raises(ClassValidationError) as caught:
        load_experiment(path)

    assert any(
        isinstance(error, ForbiddenExtraKeysError) and old_name in error.extra_fields
        for error in exception_leaves(caught.value)
    )
