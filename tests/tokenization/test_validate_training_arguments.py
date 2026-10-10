import re

import pytest

from cs336_basics.tokenization.bpe.trainer import _validate_training_arguments


@pytest.mark.parametrize(
    (
        "vocab_size",
        "special_tokens",
        "chunk_size",
        "n_process",
        "heap_after",
        "expected_error",
    ),
    [
        (
            128,
            ("<|endoftext|>",),
            1024 * 1024,
            8,
            128,
            "vocab_size must be >= 257, but got 128",
        ),
        (512, ("<|endoftext|>",), 0, 8, 128, "chunk_size must be > 0, but got 0"),
        (
            512,
            ("<|endoftext|>",),
            1024 * 1024,
            -10,
            128,
            "n_process must be > 0, but got -10",
        ),
        (
            512,
            ("<|endoftext|>",),
            1024 * 1024,
            8,
            -10,
            "heap_after must be >= 0, but got -10",
        ),
        (
            512,
            ("",),
            1024 * 1024,
            8,
            128,
            "empty string as special token is not allowed",
        ),
        (
            512,
            ("<|endoftext|>", ""),
            1024 * 1024,
            8,
            128,
            "empty string as special token is not allowed",
        ),
        (
            512,
            ("<|endoftext|>", "<|endoftext|>"),
            1024 * 1024,
            8,
            128,
            "duplicate special tokens are not allowed",
        ),
        (512, (), 1024 * 1024, 8, 128, "at least one special token is required"),
    ],
)
def test_validate_training_arguments(
    vocab_size, special_tokens, chunk_size, n_process, heap_after, expected_error
) -> None:
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        _validate_training_arguments(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            chunk_size=chunk_size,
            n_process=n_process,
            heap_after=heap_after,
        )


def test_validate_training_arguments_positive():
    _validate_training_arguments(
        vocab_size=257,
        special_tokens=("<|endoftext|>",),
        chunk_size=1,
        n_process=1,
        heap_after=0,
    )
