"""Core BPE merge algorithm.

This module owns the mutable state of BPE training. It must not parse CLI
arguments, configure logging, open output files, or create worker processes.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from .pretokenization import Pretoken, count_pretokens

Token = bytes
Merge = tuple[Token, Token]
Vocabulary = dict[int, Token]

logger = logging.getLogger(__name__)


class BPETrainer:
    """Learn BPE merges from already-counted pre-tokens.

    During implementation, keep these invariants explicit:

    * ``pair_frequencies[pair]`` equals the weighted number of adjacent
      occurrences of ``pair`` in all current words;
    * ``pair_to_word_ids[pair]`` contains exactly the words containing ``pair``;
    * a heap entry may be stale, but a selected entry must equal the current
      frequency and preserve reverse-lexicographic tie-breaking;
    * ``heap_after`` counts completed merge iterations, not vocabulary IDs.
    """

    def __init__(
        self,
        pretoken_counts: Counter[Pretoken],
        *,
        vocab_size: int,
        special_tokens: tuple[str, ...],
        heap_after: int,
    ) -> None:
        self.pretoken_counts: Counter[Pretoken] = pretoken_counts
        self.vocab_size: int = vocab_size
        self.special_tokens: tuple[str, ...] = special_tokens
        self.heap_after: int = heap_after

        self.vocabulary: Vocabulary = {token_id: bytes([token_id]) for token_id in range(256)}
        self.merges: list[Merge] = []
        self.words: dict[int, Pretoken] = {}
        self.word_frequencies: dict[int, int] = {}
        self.pair_frequencies: Counter[Merge] = Counter()
        self.pair_to_word_ids: dict[Merge, set[int]] = {}

        for word_id, (word, frequency) in enumerate(pretoken_counts.items()):
            # сохранить word и frequency
            self.words[word_id] = word
            self.word_frequencies[word_id] = frequency

            # пройти по zip(word, word[1:])
            for pair in zip(word, word[1:]):
                self.pair_frequencies[pair] += frequency

                if pair not in self.pair_to_word_ids:
                    self.pair_to_word_ids[pair] = set()
                self.pair_to_word_ids[pair].add(word_id)

    def train(self) -> tuple[Vocabulary, list[Merge]]:
        """Run merge iterations and append special tokens."""

        num_merge_slots = self.vocab_size - len(self.special_tokens) - len(self.vocabulary)

        for merge_iteration in range(num_merge_slots):
            if not self.pair_frequencies:
                break

            pair = self._select_pair(merge_iteration)
            self._apply_merge(pair)

        for special_token in self.special_tokens:
            token_id = len(self.vocabulary)
            self.vocabulary[token_id] = special_token.encode("utf-8")

        return self.vocabulary, self.merges

    def _select_pair(self, merge_iteration: int) -> Merge:
        """Choose the highest-frequency pair with the required tie-break."""
        if merge_iteration >= self.heap_after:
            raise NotImplementedError()
        else:
            # brute_force
            return max(self.pair_frequencies, key=lambda pair: (self.pair_frequencies.get(pair, 0), pair))

    def _apply_merge(self, pair: Merge) -> None:
        """Apply one non-overlapping merge and update every cache invariant."""
        self.merges.append(pair)
        self.vocabulary[len(self.vocabulary)] = pair[0] + pair[1]

        affected_word_ids = tuple(self.pair_to_word_ids[pair])

        for word_id in affected_word_ids:
            old_word: Pretoken = self.words[word_id]
            old_pairs: tuple[Merge, ...] = tuple(zip(old_word, old_word[1:]))

            word_freq = self.word_frequencies[word_id]
            for old_pair in old_pairs:
                self.pair_frequencies[old_pair] -= word_freq

                if self.pair_frequencies[old_pair] == 0:
                    del self.pair_frequencies[old_pair]

            for old_pair in set(old_pairs):
                self.pair_to_word_ids[old_pair].remove(word_id)
                if not self.pair_to_word_ids[old_pair]:
                    del self.pair_to_word_ids[old_pair]

            new_word = self._create_new_word(old_word, pair)

            for new_pair in zip(new_word[:-1], new_word[1:]):
                self.pair_frequencies[new_pair] += word_freq
                if new_pair not in self.pair_to_word_ids:
                    self.pair_to_word_ids[new_pair] = set()
                self.pair_to_word_ids[new_pair].add(word_id)

            self.words[word_id] = new_word

    def _create_new_word(
        self,
        word: tuple[bytes, ...],
        pair: Merge,
    ) -> tuple[bytes, ...]:
        new_word_parts = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word_parts.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word_parts.append(word[i])
                i += 1
        return tuple(new_word_parts)


def _validate_training_arguments(
    *,
    vocab_size: int,
    special_tokens: tuple[str, ...],
    chunk_size: int,
    n_process: int,
    heap_after: int,
) -> None:
    """Validate public API arguments before expensive work starts."""

    if vocab_size < 256 + len(special_tokens):
        raise ValueError(f"vocab_size must be >= {256 + len(special_tokens)}, but got {vocab_size}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, but got {chunk_size}")

    if n_process <= 0:
        raise ValueError(f"n_process must be > 0, but got {n_process}")

    if heap_after < 0:
        raise ValueError(f"heap_after must be >= 0, but got {heap_after}")

    if any(special_token == "" for special_token in special_tokens):
        raise ValueError("empty string as special token is not allowed")

    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("duplicate special tokens are not allowed")

    if len(special_tokens) == 0:
        raise ValueError("at least one special token is required")


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | tuple[str, ...],
    *,
    chunk_size: int = 1024 * 1024,
    n_process: int = 4,
    heap_after: int = 300,
) -> tuple[Vocabulary, list[Merge]]:
    """Train a BPE tokenizer while preserving the assignment's public API."""
    normalized_special_tokens = tuple(special_tokens)
    _validate_training_arguments(
        vocab_size=vocab_size,
        special_tokens=normalized_special_tokens,
        chunk_size=chunk_size,
        n_process=n_process,
        heap_after=heap_after,
    )
    pretoken_counts = count_pretokens(
        input_path,
        special_tokens=normalized_special_tokens,
        chunk_size=chunk_size,
        n_process=n_process,
    )
    trainer = BPETrainer(
        pretoken_counts,
        vocab_size=vocab_size,
        special_tokens=normalized_special_tokens,
        heap_after=heap_after,
    )
    return trainer.train()
