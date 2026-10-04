from __future__ import annotations

from pathlib import Path

from cs336_basics.tokenization.bpe.cli import save_tokenizer
from cs336_basics.tokenization.tokenizer import Tokenizer


def test_tokenizer_pickle_roundtrip(tmp_path: Path) -> None:
    vocabulary = {token_id: bytes([token_id]) for token_id in range(256)}
    vocabulary[256] = b"ab"
    vocabulary[257] = b"<S>"
    merges = [(b"a", b"b")]
    special_tokens = ("<S>",)
    artifact_path = tmp_path / "tokenizer.pkl"

    save_tokenizer(
        output_path=artifact_path,
        vocabulary=vocabulary,
        merges=merges,
        special_tokens=special_tokens,
    )

    tokenizer = Tokenizer.from_pickle(artifact_path)

    assert tokenizer.vocab == vocabulary
    assert tokenizer.merges == merges
    assert tokenizer.special_tokens == special_tokens
    assert tokenizer.encode("ab<S>ab") == [256, 257, 256]
    assert tokenizer.decode([256, 257, 256]) == "ab<S>ab"
