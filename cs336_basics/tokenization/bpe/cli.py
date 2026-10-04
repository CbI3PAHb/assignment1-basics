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

from .trainer import Merge, Vocabulary, train_bpe


DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"
BPE_LOGGER_NAME = "cs336_basics.tokenization.bpe"

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    TODO:
        Decide which arguments should be required and improve their help text.
        ``argv`` is injectable so the parser can be tested without patching
        ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--dataset-file-path", type=Path, required=True)
    parser.add_argument("--save-file-path", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--n-process", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--heap-after", type=int, default=300)
    parser.add_argument("--special-token", action="append", dest="special_tokens")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--json-log-file", type=Path)
    parser.add_argument("--silent", action="store_true")
    return parser.parse_args(argv)


def setup_logging(
    *,
    log_file: Path | None,
    json_log_file: Path | None,
    silent: bool,
) -> None:
    """Configure logging for one CLI process."""

    bpe_logger = logging.getLogger(BPE_LOGGER_NAME)
    bpe_logger.setLevel(logging.DEBUG)
    bpe_logger.propagate = False

    json_logger = logging.getLogger("bpe_json_logger")

    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )

    # close previously created handlers
    for handler in bpe_logger.handlers[:]:
        bpe_logger.removeHandler(handler)
        handler.close()

    # console logger
    if not silent:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)
        bpe_logger.addHandler(console_handler)
        bpe_logger.info("Console logging enabled.")

    # file logger
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
        bpe_logger.addHandler(file_handler)
        bpe_logger.info(f"Text file logging enabled. Log file: {log_file}")

    # json logger
    if json_log_file:
        log_dir = os.path.dirname(json_log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        json_logger = logging.getLogger("bpe_json_logger")
        json_logger.setLevel(logging.INFO)
        json_logger.propagate = False  # Important

        if json_logger.hasHandlers():
            json_logger.handlers.clear()

        json_file_handler = logging.FileHandler(json_log_file, mode="w", encoding="utf-8")
        json_formatter = logging.Formatter("%(message)s")
        json_file_handler.setFormatter(json_formatter)
        json_logger.addHandler(json_file_handler)
        bpe_logger.info(f"JSON logging enabled. Log file: {json_log_file}")

    if not bpe_logger.hasHandlers():
        bpe_logger.addHandler(logging.NullHandler())


def save_tokenizer(
    output_path: Path,
    vocabulary: Vocabulary,
    merges: list[Merge],
    special_tokens: tuple[str, ...],
) -> None:
    """Persist a complete tokenizer artifact atomically.

    TODO:
        Save vocabulary, merges, and special tokens to a temporary file, then
        replace ``output_path`` atomically.
    """
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

    logger.info("Starting BPE training")

    setup_logging(
        log_file=args.log_file,
        json_log_file=args.json_log_file,
        silent=args.silent,
    )
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
