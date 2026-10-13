"""Command-line interface for BPE training.

Keep filesystem output, argument parsing, and application-level logging here.
The training implementation must not depend on ``argparse.Namespace``.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from collections.abc import Sequence
from pathlib import Path

from cs336_basics.logging_setup import setup_logging

from .trainer import Merge, Vocabulary, train_bpe

DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--dataset-file-path", type=Path, required=True)
    parser.add_argument("--save-file-path", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--n-process", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--heap-after", type=int, default=300)
    parser.add_argument("--special-token", action="append", dest="special_tokens")
    parser.add_argument("--silent", action="store_true")
    return parser.parse_args(argv)


def save_tokenizer(
    output_path: Path,
    vocabulary: Vocabulary,
    merges: list[Merge],
    special_tokens: tuple[str, ...],
) -> None:
    """Persist a complete tokenizer artifact atomically"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")

    payload = {"vocab": vocabulary, "merges": merges, "special_tokens": special_tokens}

    try:
        with temp_path.open("wb") as temp_file:
            pickle.dump(payload, temp_file)

        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)

    logger.info("Saved tokenizer to %s", output_path)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI pipeline: parse, configure, train, and save."""
    args = parse_args(argv)
    special_tokens = tuple(args.special_tokens or [DEFAULT_SPECIAL_TOKEN])

    setup_logging(silent=args.silent)
    logger.info("Starting BPE training")
    vocabulary, merges = train_bpe(
        input_path=args.dataset_file_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        chunk_size=args.chunk_size,
        n_process=args.n_process,
        heap_after=args.heap_after,
    )
    save_tokenizer(
        output_path=args.save_file_path,
        vocabulary=vocabulary,
        merges=merges,
        special_tokens=special_tokens,
    )

    logger.info("BPE training done!")
