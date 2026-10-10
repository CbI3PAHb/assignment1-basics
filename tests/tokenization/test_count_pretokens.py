from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

import cs336_basics.tokenization.bpe.pretokenization as pretokenization
from cs336_basics.tokenization.bpe.pretokenization import count_pretokens

SPECIAL_TOKENS = ("<|endoftext|>",)


def _byte_tokens(text: str) -> tuple[bytes, ...]:
    return tuple(bytes([value]) for value in text.encode("utf-8"))


@pytest.mark.parametrize(
    ("chunk_size", "n_process"),
    [
        pytest.param(1, 1, id="one-worker"),
        pytest.param(10_000, 8, id="one-task"),
    ],
)
def test_count_pretokens_uses_serial_fast_path(
    tmp_path: Path,
    chunk_size: int,
    n_process: int,
) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(
        "foo!foo<|endoftext|>bar",
        encoding="utf-8",
    )

    counts = count_pretokens(
        str(input_path),
        special_tokens=SPECIAL_TOKENS,
        chunk_size=chunk_size,
        n_process=n_process,
    )

    assert counts == Counter(
        {
            _byte_tokens("foo"): 2,
            _byte_tokens("!"): 1,
            _byte_tokens("bar"): 1,
        }
    )


def test_count_pretokens_accepts_empty_file(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.txt"
    input_path.write_bytes(b"")

    counts = count_pretokens(
        input_path,
        special_tokens=SPECIAL_TOKENS,
        chunk_size=10,
        n_process=1,
    )

    assert counts == Counter()


def test_count_pretokens_rejects_non_positive_process_count(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("text", encoding="utf-8")

    with pytest.raises(ValueError, match="n_process must be > 0"):
        count_pretokens(
            input_path,
            special_tokens=SPECIAL_TOKENS,
            chunk_size=10,
            n_process=0,
        )


def test_count_pretokens_rejects_non_positive_chunk_size(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("text", encoding="utf-8")

    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        count_pretokens(
            input_path,
            special_tokens=SPECIAL_TOKENS,
            chunk_size=0,
            n_process=1,
        )


def test_count_pretokens_limits_workers_to_actual_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(
        "left<|endoftext|>right",
        encoding="utf-8",
    )
    pool_sizes: list[int] = []

    class RecordingPool:
        def __init__(self, processes: int) -> None:
            pool_sizes.append(processes)

        def __enter__(self) -> RecordingPool:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def imap_unordered(self, function, tasks):
            return map(function, tasks)

    monkeypatch.setattr(pretokenization.multiprocessing, "Pool", RecordingPool)

    counts = count_pretokens(
        input_path,
        special_tokens=SPECIAL_TOKENS,
        chunk_size=1,
        n_process=8,
    )

    assert pool_sizes == [2]
    assert counts == Counter(
        {
            _byte_tokens("left"): 1,
            _byte_tokens("right"): 1,
        }
    )
