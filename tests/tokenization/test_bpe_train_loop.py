from __future__ import annotations

from collections import Counter

from cs336_basics.tokenization.bpe.trainer import BPETrainer


def test_train_runs_brute_force_merges_until_target_vocab_size() -> None:
    trainer = BPETrainer(
        Counter({(b"a", b"b", b"a", b"b"): 1}),
        vocab_size=259,
        special_tokens=("<S>",),
        heap_after=10,
    )

    vocabulary, merges = trainer.train()

    assert merges == [
        (b"a", b"b"),
        (b"ab", b"ab"),
    ]
    assert vocabulary[256] == b"ab"
    assert vocabulary[257] == b"abab"
    assert vocabulary[258] == b"<S>"
    assert len(vocabulary) == 259


def test_train_does_not_merge_when_only_special_token_slot_remains() -> None:
    trainer = BPETrainer(
        Counter({(b"a", b"b"): 10}),
        vocab_size=257,
        special_tokens=("<S>",),
        heap_after=10,
    )

    vocabulary, merges = trainer.train()

    assert merges == []
    assert vocabulary[256] == b"<S>"
    assert len(vocabulary) == 257


def test_train_stops_without_pairs_and_still_adds_special_tokens() -> None:
    trainer = BPETrainer(
        Counter({(b"a",): 5}),
        vocab_size=300,
        special_tokens=("<FIRST>", "<SECOND>"),
        heap_after=10,
    )

    vocabulary, merges = trainer.train()

    assert merges == []
    assert vocabulary[256] == b"<FIRST>"
    assert vocabulary[257] == b"<SECOND>"
    assert len(vocabulary) == 258
