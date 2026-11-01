from __future__ import annotations

from pathlib import Path

import pytest

from cs336_basics.tokenization.bpe.pretokenization import find_safe_chunk_boundaries

SPLIT_TOKEN = b"<S>"


def _write_corpus(tmp_path: Path, contents: bytes) -> Path:
    input_path = tmp_path / "corpus.txt"
    input_path.write_bytes(contents)
    return input_path


def test_find_safe_chunk_boundaries_uses_custom_split_token(
    tmp_path: Path,
) -> None:
    contents = b"a" * 15 + SPLIT_TOKEN + b"b" * 12 + SPLIT_TOKEN + b"c" * 12 + SPLIT_TOKEN + b"d" * 12
    input_path = _write_corpus(tmp_path, contents)

    boundaries = find_safe_chunk_boundaries(
        input_path,
        desired_num_chunks=4,
        split_token=SPLIT_TOKEN,
    )

    assert boundaries == [0, 15, 30, 45, len(contents)]
    assert all(contents.startswith(SPLIT_TOKEN, boundary) for boundary in boundaries[1:-1])


def test_find_safe_chunk_boundaries_returns_fewer_chunks_without_delimiters(
    tmp_path: Path,
) -> None:
    contents = b"there are no document delimiters here"
    input_path = _write_corpus(tmp_path, contents)

    boundaries = find_safe_chunk_boundaries(
        input_path,
        desired_num_chunks=8,
        split_token=SPLIT_TOKEN,
    )

    assert boundaries == [0, len(contents)]


def test_find_safe_chunk_boundaries_finds_token_across_read_blocks(
    tmp_path: Path,
) -> None:
    file_size = 10_000
    first_search_position = file_size // 2
    token_position = first_search_position + 4096
    contents = b"a" * token_position + SPLIT_TOKEN + b"b" * (file_size - token_position - len(SPLIT_TOKEN))
    input_path = _write_corpus(tmp_path, contents)

    boundaries = find_safe_chunk_boundaries(
        input_path,
        desired_num_chunks=2,
        split_token=SPLIT_TOKEN,
    )

    assert boundaries == [0, token_position, file_size]


def test_find_safe_chunk_boundaries_returns_one_requested_chunk(
    tmp_path: Path,
) -> None:
    contents = b"left<S>right"
    input_path = _write_corpus(tmp_path, contents)

    boundaries = find_safe_chunk_boundaries(
        input_path,
        desired_num_chunks=1,
        split_token=SPLIT_TOKEN,
    )

    assert boundaries == [0, len(contents)]


def test_find_safe_chunk_boundaries_represents_empty_file(
    tmp_path: Path,
) -> None:
    input_path = _write_corpus(tmp_path, b"")

    boundaries = find_safe_chunk_boundaries(
        input_path,
        desired_num_chunks=4,
        split_token=SPLIT_TOKEN,
    )

    assert boundaries == [0, 0]


@pytest.mark.parametrize("desired_num_chunks", [0, -1])
def test_find_safe_chunk_boundaries_rejects_non_positive_chunk_count(
    tmp_path: Path,
    desired_num_chunks: int,
) -> None:
    input_path = _write_corpus(tmp_path, b"text")

    with pytest.raises(ValueError, match="desired_num_chunks must be > 0"):
        find_safe_chunk_boundaries(
            input_path,
            desired_num_chunks=desired_num_chunks,
            split_token=SPLIT_TOKEN,
        )


def test_find_safe_chunk_boundaries_rejects_empty_split_token(
    tmp_path: Path,
) -> None:
    input_path = _write_corpus(tmp_path, b"text")

    with pytest.raises(
        ValueError,
        match="split_token must not be empty",
    ):
        find_safe_chunk_boundaries(
            input_path,
            desired_num_chunks=2,
            split_token=b"",
        )
