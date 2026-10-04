"""
uv run python -m cs336_basics.tokenization.encode_dataset \
    --tokenizer-path artifacts/tokenizers/tinystories/tokenizer.pkl \
    --input-path data/TinyStoriesV2-GPT4-train.txt \
    --output-path tokenized_datasets/tinystories_train.bin
"""
import os
import numpy as np

from collections.abc import Iterator
from typing import TextIO
from cs336_basics.tokenization.tokenizer import Tokenizer
from pathlib import Path


def iter_documents(
    input_file: TextIO,
    document_separator: str,
    read_size: int = 1024 * 1024,
) -> Iterator[str]:
    remainder = ""

    while chunk := input_file.read(read_size):
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
    write_buffer_size: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        raise RuntimeError(f"{output_path} already existed")

    token_count = 0
    token_buffer: list[int] = []

    with open(input_path, "r", encoding='utf-8') as input_file:
        with open(output_path, "wb") as output_file:
            documents = iter_documents(input_file, document_separator)
            token_ids = tokenizer.encode_iterable(documents)
                
            for token_id in token_ids:
                token_buffer.append(token_id)
                token_count += 1
                if len(token_buffer) >= write_buffer_size:
                    np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)
                    token_buffer.clear()

            # last token buffer
            if token_buffer:
                np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)

    return token_count
