from __future__ import annotations

import pickle
from pathlib import Path
from typing import cast

import pytest

from cs336_basics.tokenization.bpe.cli import save_tokenizer
from cs336_basics.tokenization.bpe.trainer import Vocabulary


def test_save_tokenizer_writes_complete_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "tokenizer.pkl"
    vocabulary = {0: b"a", 1: b"b", 2: b"ab", 3: b"<S>"}
    merges = [(b"a", b"b")]
    special_tokens = ("<S>",)

    save_tokenizer(
        output_path=output_path,
        vocabulary=vocabulary,
        merges=merges,
        special_tokens=special_tokens,
    )

    with output_path.open("rb") as artifact_file:
        artifact = pickle.load(artifact_file)

    assert artifact == {
        "vocab": vocabulary,
        "merges": merges,
        "special_tokens": special_tokens,
    }


class _UnpicklableToken:
    def __reduce__(self) -> object:
        raise RuntimeError("deliberate serialization failure")


def test_save_tokenizer_preserves_existing_artifact_when_serialization_fails(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "tokenizer.pkl"
    original_contents = b"existing tokenizer artifact"
    output_path.write_bytes(original_contents)
    invalid_vocabulary = cast(Vocabulary, {0: _UnpicklableToken()})

    with pytest.raises(RuntimeError, match="deliberate serialization failure"):
        save_tokenizer(
            output_path=output_path,
            vocabulary=invalid_vocabulary,
            merges=[],
            special_tokens=("<S>",),
        )

    assert output_path.read_bytes() == original_contents
    assert list(tmp_path.iterdir()) == [output_path]
