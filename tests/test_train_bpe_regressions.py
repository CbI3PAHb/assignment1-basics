from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import regex as re

from cs336_basics.pretokenization_example import PAT
from cs336_basics.tokenization.bpe import train_bpe


def _merge_pair(
    tokens: tuple[bytes, ...], pair_to_merge: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    """Apply one BPE merge from left to right without overlapping merges."""
    merged_tokens: list[bytes] = []
    index = 0

    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and (tokens[index], tokens[index + 1]) == pair_to_merge
        ):
            merged_tokens.append(tokens[index] + tokens[index + 1])
            index += 2
        else:
            merged_tokens.append(tokens[index])
            index += 1

    return tuple(merged_tokens)


def _reference_train_bpe(
    text: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Small, deliberately slow BPE oracle used only by regression tests."""
    vocabulary = {token_id: bytes([token_id]) for token_id in range(256)}

    if special_tokens:
        special_token_pattern = "|".join(
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        text_parts = re.split(special_token_pattern, text)
    else:
        text_parts = [text]

    word_frequencies: Counter[tuple[bytes, ...]] = Counter()
    for text_part in text_parts:
        for match in re.finditer(PAT, text_part):
            word = tuple(bytes([byte]) for byte in match.group().encode("utf-8"))
            word_frequencies[word] += 1

    merges: list[tuple[bytes, bytes]] = []
    num_learned_tokens = vocab_size - len(special_tokens)

    for new_token_id in range(256, num_learned_tokens):
        pair_frequencies: Counter[tuple[bytes, bytes]] = Counter()
        for word, frequency in word_frequencies.items():
            for pair in zip(word, word[1:]):
                pair_frequencies[pair] += frequency

        if not pair_frequencies:
            break

        pair_to_merge = max(
            pair_frequencies,
            key=lambda pair: (pair_frequencies[pair], pair),
        )
        merges.append(pair_to_merge)
        vocabulary[new_token_id] = pair_to_merge[0] + pair_to_merge[1]

        merged_word_frequencies: Counter[tuple[bytes, ...]] = Counter()
        for word, frequency in word_frequencies.items():
            merged_word_frequencies[_merge_pair(word, pair_to_merge)] += frequency
        word_frequencies = merged_word_frequencies

    for special_token in special_tokens:
        vocabulary[len(vocabulary)] = special_token.encode("utf-8")

    return vocabulary, merges


def _write_corpus(tmp_path: Path, text: str) -> Path:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(text, encoding="utf-8")
    return corpus_path


def test_brute_force_training_matches_reference(tmp_path: Path) -> None:
    text = "becfebabeeeabacfbcbfffeccbcbbcebadbfcebbcfdcfedcedefdfac"
    corpus_path = _write_corpus(tmp_path, text)

    expected = _reference_train_bpe(text, 330, ["<|endoftext|>"])
    actual = train_bpe(
        corpus_path,
        vocab_size=330,
        special_tokens=["<|endoftext|>"],
        chunk_size=1_000_000,
        n_process=1,
        heap_after=1_000,
    )

    assert actual == expected


def test_overlapping_special_tokens_are_order_independent(tmp_path: Path) -> None:
    text = "<A>xq"
    corpus_path = _write_corpus(tmp_path, text)
    short_first = ["<A>", "<A>x"]
    long_first = list(reversed(short_first))

    _, expected_merges = _reference_train_bpe(text, 259, short_first)
    _, merges_with_short_first = train_bpe(
        corpus_path,
        vocab_size=259,
        special_tokens=short_first,
        n_process=1,
    )
    _, merges_with_long_first = train_bpe(
        corpus_path,
        vocab_size=259,
        special_tokens=long_first,
        n_process=1,
    )

    assert merges_with_short_first == expected_merges
    assert merges_with_long_first == expected_merges


def test_chunk_size_does_not_change_merges_for_custom_special_token(
    tmp_path: Path,
) -> None:
    text = "a <|endoftext|> b a <|endoftext|> b"
    corpus_path = _write_corpus(tmp_path, text)
    special_tokens = ["<CUSTOM>"]

    expected = _reference_train_bpe(text, 275, special_tokens)
    one_chunk = train_bpe(
        corpus_path,
        vocab_size=275,
        special_tokens=special_tokens,
        chunk_size=10_000,
        n_process=1,
        heap_after=1_000,
    )
    many_chunks = train_bpe(
        corpus_path,
        vocab_size=275,
        special_tokens=special_tokens,
        chunk_size=1,
        n_process=1,
        heap_after=1_000,
    )

    assert one_chunk == expected
    assert many_chunks == expected


def test_empty_corpus_returns_base_vocabulary(tmp_path: Path) -> None:
    corpus_path = _write_corpus(tmp_path, "")

    expected = _reference_train_bpe("", 257, ["<|endoftext|>"])
    actual = train_bpe(
        corpus_path,
        vocab_size=257,
        special_tokens=["<|endoftext|>"],
        n_process=1,
    )

    assert actual == expected


def test_vocab_size_must_include_bytes_and_special_tokens(tmp_path: Path) -> None:
    corpus_path = _write_corpus(tmp_path, "abc")

    with pytest.raises(ValueError):
        train_bpe(
            corpus_path,
            vocab_size=256,
            special_tokens=["<|endoftext|>"],
            n_process=1,
        )


def test_chunk_size_must_be_positive(tmp_path: Path) -> None:
    corpus_path = _write_corpus(tmp_path, "abc")

    with pytest.raises(ValueError):
        train_bpe(
            corpus_path,
            vocab_size=260,
            special_tokens=["<|endoftext|>"],
            chunk_size=0,
            n_process=1,
        )
