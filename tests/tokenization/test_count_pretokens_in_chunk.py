from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from cs336_basics.tokenization.bpe.pretokenization import ChunkTask, _count_pretokens_in_chunk

SPECIAL_TOKENS = ("<|endoftext|>",)


def _byte_tokens(text: str) -> tuple[bytes, ...]:
    return tuple(bytes([value]) for value in text.encode("utf-8"))


def _write_bytes(tmp_path: Path, contents: bytes) -> Path:
    input_path = tmp_path / "corpus.txt"
    input_path.write_bytes(contents)
    return input_path


def test_count_pretokens_counts_repeated_tokens(tmp_path: Path) -> None:
    contents = b"foo!foo!foo"
    input_path = _write_bytes(tmp_path, contents)
    task = ChunkTask(input_path, 0, len(contents), SPECIAL_TOKENS)

    counts = _count_pretokens_in_chunk(task)

    assert counts == Counter({_byte_tokens("foo"): 3, _byte_tokens("!"): 2})


def test_count_pretokens_reads_exact_nonzero_utf8_range(tmp_path: Path) -> None:
    prefix = "discard|".encode("utf-8")
    selected = "é🙂é🙂".encode("utf-8")
    suffix = "|discard".encode("utf-8")
    input_path = _write_bytes(tmp_path, prefix + selected + suffix)
    task = ChunkTask(
        input_path,
        len(prefix),
        len(prefix) + len(selected),
        SPECIAL_TOKENS,
    )

    counts = _count_pretokens_in_chunk(task)

    assert counts == Counter({_byte_tokens("é"): 2, _byte_tokens("🙂"): 2})


def test_count_pretokens_accepts_empty_range(tmp_path: Path) -> None:
    input_path = _write_bytes(tmp_path, b"")
    task = ChunkTask(input_path, 0, 0, SPECIAL_TOKENS)

    assert _count_pretokens_in_chunk(task) == Counter()


def test_count_pretokens_treats_special_token_as_separator(tmp_path: Path) -> None:
    contents = b"left<|endoftext|>right"
    input_path = _write_bytes(tmp_path, contents)
    task = ChunkTask(input_path, 0, len(contents), ("<|endoftext|>",))

    counts = _count_pretokens_in_chunk(task)

    assert counts == Counter({_byte_tokens("left"): 1, _byte_tokens("right"): 1})


@pytest.mark.parametrize(
    "special_tokens",
    [
        pytest.param(("<A>", "<A>x"), id="short-token-first"),
        pytest.param(("<A>x", "<A>"), id="long-token-first"),
    ],
)
def test_count_pretokens_handles_overlapping_special_tokens_longest_first(
    tmp_path: Path,
    special_tokens: tuple[str, ...],
) -> None:
    contents = b"<A>xq<A>z"
    input_path = _write_bytes(tmp_path, contents)
    task = ChunkTask(input_path, 0, len(contents), special_tokens)

    counts = _count_pretokens_in_chunk(task)

    assert counts == Counter({_byte_tokens("q"): 1, _byte_tokens("z"): 1})


def test_count_pretokens_rejects_range_that_splits_utf8_codepoint(
    tmp_path: Path,
) -> None:
    contents = "é".encode("utf-8")
    input_path = _write_bytes(tmp_path, contents)
    task = ChunkTask(input_path, 1, len(contents), SPECIAL_TOKENS)

    with pytest.raises(UnicodeDecodeError):
        _count_pretokens_in_chunk(task)
