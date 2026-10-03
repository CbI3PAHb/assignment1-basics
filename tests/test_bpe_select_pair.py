from __future__ import annotations

from collections import Counter

import pytest

from cs336_basics.tokenization.bpe.trainer import BPETrainer, Merge


@pytest.mark.parametrize(
    ("pretoken_counts", "expected_pair"),
    [
        pytest.param(
            Counter(
                {
                    (b"a", b"b"): 5,
                    (b"z", b"a"): 3,
                }
            ),
            (b"a", b"b"),
            id="highest-frequency",
        ),
        pytest.param(
            Counter(
                {
                    (b"a", b"b"): 2,
                    (b"a", b"c"): 2,
                }
            ),
            (b"a", b"c"),
            id="lexicographically-largest-tie-break",
        ),
    ],
)
def test_select_pair_brute_force(
    pretoken_counts: Counter[tuple[bytes, ...]],
    expected_pair: Merge,
) -> None:
    trainer = BPETrainer(
        pretoken_counts,
        vocab_size=270,
        special_tokens=("<|endoftext|>",),
        heap_after=10,
    )

    selected_pair = trainer._select_pair(merge_iteration=0)

    assert selected_pair == expected_pair
