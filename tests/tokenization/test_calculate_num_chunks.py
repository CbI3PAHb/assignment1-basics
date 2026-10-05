from __future__ import annotations

import pytest

from cs336_basics.tokenization.bpe.pretokenization import calculate_num_chunks


@pytest.mark.parametrize(
    ("file_size", "chunk_size", "expected_num_chunks"),
    [
        pytest.param(0, 10, 1, id="empty-file"),
        pytest.param(1, 10, 1, id="smaller-than-one-chunk"),
        pytest.param(10, 10, 1, id="exactly-one-chunk"),
        pytest.param(11, 10, 2, id="one-byte-over"),
        pytest.param(20, 10, 2, id="exact-multiple"),
        pytest.param(21, 10, 3, id="exact-multiple-plus-one"),
        pytest.param(10**18 + 1, 10**18, 2, id="large-integers"),
    ],
)
def test_calculate_num_chunks_rounds_up(
    file_size: int,
    chunk_size: int,
    expected_num_chunks: int,
) -> None:
    assert calculate_num_chunks(file_size, chunk_size) == expected_num_chunks


def test_calculate_num_chunks_rejects_negative_file_size() -> None:
    with pytest.raises(ValueError, match="file_size must be >= 0"):
        calculate_num_chunks(file_size=-1, chunk_size=10)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_calculate_num_chunks_rejects_non_positive_chunk_size(
    chunk_size: int,
) -> None:
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        calculate_num_chunks(file_size=10, chunk_size=chunk_size)
