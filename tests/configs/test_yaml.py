import pytest
import yaml
from yaml.constructor import ConstructorError

from cs336_basics.training.config._yaml import _UniqueKeyLoader


def _load(text: str):
    return yaml.load(text, Loader=_UniqueKeyLoader)


@pytest.mark.parametrize(
    ("text", "first_location", "duplicate_location"),
    [
        ("name: same\nname: same\n", (0, 0), (1, 0)),
        ("name: first\nname: second\n", (0, 0), (1, 0)),
        ("outer:\n  name: first\n  name: second\n", (1, 2), (2, 2)),
        ("{name: first, name: second}", (0, 1), (0, 14)),
        ("- name: first\n  name: second\n", (0, 2), (1, 2)),
    ],
)
def test_duplicates_are_rejected_with_both_source_locations(text, first_location, duplicate_location):
    with pytest.raises(ConstructorError, match="found duplicate key 'name'") as caught:
        _load(text)

    error = caught.value
    assert error.context == "first occurrence of this key"
    assert (error.context_mark.line, error.context_mark.column) == first_location
    assert (error.problem_mark.line, error.problem_mark.column) == duplicate_location


def test_repeated_key_names_in_separate_mappings_are_allowed():
    parsed = _load("left: {name: first}\nright: {name: second}\nrows: [{name: third}, {name: fourth}]\n")

    assert parsed == {
        "left": {"name": "first"},
        "right": {"name": "second"},
        "rows": [{"name": "third"}, {"name": "fourth"}],
    }


def test_aliases_preserve_shared_objects_and_recursive_mappings():
    parsed = _load(
        "shared: &shared {name: value}\n"
        "first: *shared\n"
        "second: *shared\n"
        "recursive: &recursive\n"
        "  self: *recursive\n"
    )

    assert parsed["first"] is parsed["shared"]
    assert parsed["second"] is parsed["shared"]
    assert parsed["recursive"]["self"] is parsed["recursive"]


def test_merges_of_distinct_keys_are_allowed():
    parsed = _load(
        "left: &left {a: 1}\n"
        "right: &right {b: 2}\n"
        "combined:\n"
        "  <<: [*left, *right]\n"
        "  c: 3\n"
    )

    assert parsed["combined"] == {"a": 1, "b": 2, "c": 3}


@pytest.mark.parametrize(
    "text",
    [
        "base: &base {name: inherited}\nmerged:\n  <<: *base\n  name: explicit\n",
        "base: &base {name: same}\nmerged:\n  <<: *base\n  name: same\n",
        "left: &left {name: first}\nright: &right {name: second}\nmerged: {<<: [*left, *right]}\n",
    ],
)
def test_merge_collisions_are_rejected_even_for_identical_values(text):
    with pytest.raises(ConstructorError, match="found duplicate key 'name'"):
        _load(text)


def test_merge_collision_marks_point_to_inherited_and_explicit_keys():
    with pytest.raises(ConstructorError) as caught:
        _load("base: &base {name: inherited}\nmerged:\n  <<: *base\n  name: explicit\n")

    assert caught.value.context_mark.line == 0
    assert caught.value.problem_mark.line == 3


def test_unhashable_mapping_key_reports_constructor_error():
    with pytest.raises(ConstructorError, match="found unhashable key"):
        _load("? [one, two]\n: value\n")


def test_empty_documents_and_scalar_parsing_are_unchanged():
    assert _load("") is None
    assert _load("42") == 42

    parsed = _load("integer: 42\nfloat: 0.25\nboolean: true\ntext: '42'\nmissing: null\nsequence: [1, two]\n")
    assert parsed == {
        "integer": 42,
        "float": 0.25,
        "boolean": True,
        "text": "42",
        "missing": None,
        "sequence": [1, "two"],
    }
    assert type(parsed["integer"]) is int
    assert type(parsed["float"]) is float
    assert type(parsed["boolean"]) is bool


@pytest.mark.parametrize("text", ["model: [1, 2", "first: 1\n---\nsecond: 2\n"])
def test_malformed_yaml_and_multiple_documents_are_rejected(text):
    with pytest.raises(yaml.YAMLError):
        _load(text)


def test_strict_loader_does_not_change_global_safe_loader_behavior():
    original_construct_mapping = yaml.SafeLoader.construct_mapping
    original_constructors = yaml.SafeLoader.yaml_constructors.copy()
    duplicate_text = "name: first\nname: second\n"

    with pytest.raises(ConstructorError):
        _load(duplicate_text)

    assert yaml.SafeLoader.construct_mapping is original_construct_mapping
    assert yaml.SafeLoader.yaml_constructors == original_constructors
    assert yaml.safe_load(duplicate_text) == {"name": "second"}


def test_python_object_construction_tags_are_rejected():
    with pytest.raises(ConstructorError, match="could not determine a constructor"):
        _load("!!python/object/apply:builtins.dict []")
