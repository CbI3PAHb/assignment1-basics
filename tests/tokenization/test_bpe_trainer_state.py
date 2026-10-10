from __future__ import annotations

from collections import Counter

from cs336_basics.tokenization.bpe.trainer import BPETrainer


def test_bpe_trainer_initializes_word_and_pair_indexes() -> None:
    repeated_pair_word = (b"a", b"a", b"a")
    mixed_pair_word = (b"a", b"a", b"b", b"a")
    single_token_word = (b"x",)
    pretoken_counts = Counter(
        {
            repeated_pair_word: 2,
            mixed_pair_word: 3,
            single_token_word: 7,
        }
    )

    trainer = BPETrainer(
        pretoken_counts,
        vocab_size=270,
        special_tokens=("<|endoftext|>",),
        heap_after=10,
    )

    assert trainer.vocabulary == {token_id: bytes([token_id]) for token_id in range(256)}
    assert trainer.merges == []

    word_ids = {word: word_id for word_id, word in trainer.words.items()}
    assert set(trainer.words) == set(range(len(pretoken_counts)))
    assert set(word_ids) == {
        repeated_pair_word,
        mixed_pair_word,
        single_token_word,
    }
    assert trainer.word_frequencies == {
        word_ids[repeated_pair_word]: 2,
        word_ids[mixed_pair_word]: 3,
        word_ids[single_token_word]: 7,
    }

    assert trainer.pair_frequencies == Counter(
        {
            (b"a", b"a"): 7,
            (b"a", b"b"): 3,
            (b"b", b"a"): 3,
        }
    )
    assert trainer.pair_to_word_ids == {
        (b"a", b"a"): {
            word_ids[repeated_pair_word],
            word_ids[mixed_pair_word],
        },
        (b"a", b"b"): {word_ids[mixed_pair_word]},
        (b"b", b"a"): {word_ids[mixed_pair_word]},
    }
