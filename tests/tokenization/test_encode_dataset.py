from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np

from cs336_basics.tokenization.encode_dataset import encode_dataset, iter_documents
from cs336_basics.tokenization.tokenizer import Tokenizer


def test_iter_documents_is_independent_of_read_size() -> None:
    text = "first<S>second<S>tail"
    expected_documents = ["first<S>", "second<S>", "tail"]

    for read_size in (1, 2, 3, 4, 1024):
        documents = iter_documents(
            StringIO(text),
            document_separator="<S>",
            read_size=read_size,
        )

        assert list(documents) == expected_documents


def test_encode_dataset_writes_streamed_uint16_tokens(tmp_path: Path) -> None:
    vocabulary = {token_id: bytes([token_id]) for token_id in range(256)}
    vocabulary[256] = b"ab"
    vocabulary[257] = b"<S>"
    tokenizer = Tokenizer(
        vocab=vocabulary,
        merges=[(b"a", b"b")],
        special_tokens=["<S>"],
    )
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "nested" / "tokens.bin"
    input_path.write_text("ab<S>ab\nab", encoding="utf-8")

    token_count = encode_dataset(
        tokenizer=tokenizer,
        input_path=input_path,
        output_path=output_path,
        document_separator="<S>",
        write_buffer_size=2,
    )

    token_ids = np.fromfile(output_path, dtype=np.uint16)

    assert token_count == 5
    assert token_ids.tolist() == [256, 257, 256, 10, 256]
