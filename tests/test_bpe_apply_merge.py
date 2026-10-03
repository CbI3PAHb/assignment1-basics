from __future__ import annotations

from collections import Counter

from cs336_basics.tokenization.bpe.trainer import BPETrainer


def test_apply_merge_updates_words_and_pair_indexes() -> None:
    overlapping_word = (b"a", b"a", b"a")
    surrounded_word = (b"b", b"a", b"a", b"c")
    unaffected_word = (b"x", b"y")
    trainer = BPETrainer(
        Counter(
            {
                overlapping_word: 2,
                surrounded_word: 3,
                unaffected_word: 5,
            }
        ),
        vocab_size=270,
        special_tokens=("<|endoftext|>",),
        heap_after=10,
    )
    word_ids = {word: word_id for word_id, word in trainer.words.items()}
    vocabulary_before_merge = trainer.vocabulary.copy()

    trainer._apply_merge((b"a", b"a"))

    overlapping_word_id = word_ids[overlapping_word]
    surrounded_word_id = word_ids[surrounded_word]
    unaffected_word_id = word_ids[unaffected_word]
    assert trainer.words == {
        overlapping_word_id: (b"aa", b"a"),
        surrounded_word_id: (b"b", b"aa", b"c"),
        unaffected_word_id: unaffected_word,
    }
    assert trainer.word_frequencies == {
        overlapping_word_id: 2,
        surrounded_word_id: 3,
        unaffected_word_id: 5,
    }
    assert trainer.pair_frequencies == Counter(
        {
            (b"aa", b"a"): 2,
            (b"b", b"aa"): 3,
            (b"aa", b"c"): 3,
            (b"x", b"y"): 5,
        }
    )
    assert trainer.pair_to_word_ids == {
        (b"aa", b"a"): {overlapping_word_id},
        (b"b", b"aa"): {surrounded_word_id},
        (b"aa", b"c"): {surrounded_word_id},
        (b"x", b"y"): {unaffected_word_id},
    }
    assert trainer.vocabulary == {
        **vocabulary_before_merge,
        256: b"aa",
    }
    assert trainer.merges == [(b"a", b"a")]


def test_apply_merge_compares_pair_at_current_position() -> None:
    word = (b"a", b"b", b"a", b"c")
    trainer = BPETrainer(
        Counter({word: 1}),
        vocab_size=270,
        special_tokens=("<|endoftext|>",),
        heap_after=10,
    )
    word_id = next(iter(trainer.words))

    trainer._apply_merge((b"a", b"b"))

    assert trainer.words[word_id] == (b"ab", b"a", b"c")
    assert trainer.pair_frequencies == Counter(
        {
            (b"ab", b"a"): 1,
            (b"a", b"c"): 1,
        }
    )
    assert trainer.pair_to_word_ids == {
        (b"ab", b"a"): {word_id},
        (b"a", b"c"): {word_id},
    }
