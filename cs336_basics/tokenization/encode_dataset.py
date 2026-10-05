"""
uv run python -m cs336_basics.tokenization.encode_dataset \
    --input-path /home/parii-artem/wip/EFDL/assignment1-basics/data/owt_valid.txt \
    --output-path tokenized_datasets/owt_valid.bin \
    --tokenizer-file-path artifacts/tokenizers/owt_valid/tokenizer.pkl
"""

import numpy as np
import argparse
import os
from tqdm import tqdm

from collections.abc import Iterator, Sequence
from typing import TextIO
from cs336_basics.tokenization.tokenizer import Tokenizer
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def iter_documents(
    input_file: TextIO,
    document_separator: str,
    read_size: int = 1024 * 1024,
) -> Iterator[str]:
    remainder = ""

    input_file.seek(0, os.SEEK_END)
    file_size = input_file.tell()
    input_file.seek(0)
        
    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Reading") as pbar:
        while chunk := input_file.read(read_size):
            pbar.update(len(chunk))

            remainder += chunk
            parts = remainder.split(document_separator)

            remainder = parts.pop()

            for document in parts:
                yield document + document_separator

        if remainder:
            yield remainder


def encode_dataset(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    *,
    document_separator: str,
    write_buffer_tokens: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if write_buffer_tokens <= 0:
        raise ValueError("write_buffer_tokens must be positive")

    if output_path.exists():
        raise RuntimeError(f"{output_path} already exists")

    if document_separator not in tokenizer.special_tokens:
        raise ValueError(f"Document separator {document_separator!r} must be a tokenizer special token")

    if tokenizer.vocab_size - 1 > np.iinfo(np.uint16).max:
        raise RuntimeError(f"got {tokenizer.vocab_size}")

    temp_path = output_path.with_name(f".{output_path.name}.tmp")

    try:
        with input_path.open(encoding="utf-8") as input_file, temp_path.open("wb") as output_file:
            token_count = 0
            token_buffer: list[int] = []

            documents = iter_documents(input_file, document_separator, read_size=2 * 1024 * 1024)
            token_ids = tokenizer.encode_iterable(documents)

            for token_id in token_ids:
                token_buffer.append(token_id)
                token_count += 1
                if len(token_buffer) >= write_buffer_tokens:
                    np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)
                    token_buffer.clear()

            # last token buffer
            if token_buffer:
                np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)

        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return token_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(description="Encode dataset")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--tokenizer-file-path", type=Path, required=True)
    parser.add_argument("--document-separator", type=str, default="<|endoftext|>")
    parser.add_argument("--write-buffer-tokens", type=int, default=1024 * 1024)  # 1M tokens
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI pipeline: parse, configure, train, and save."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(f"Encoding dataset from {args.input_path}")
    tokenizer = Tokenizer.from_pickle(args.tokenizer_file_path)

    tokens_count = encode_dataset(
        tokenizer,
        args.input_path,
        args.output_path,
        document_separator=args.document_separator,
        write_buffer_tokens=args.write_buffer_tokens,
    )
    logger.info(f"{tokens_count=}")


if __name__ == "__main__":
    main()
