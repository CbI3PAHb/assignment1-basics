import json
from typing import Iterable, Iterator, Type

import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization_example import PAT

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pair_ranks: dict[tuple[bytes, bytes], int] = self._prepare_pair_ranks()

    def _prepare_pair_ranks(self) -> dict[tuple[bytes, bytes], int]:
        pair_ranks: dict[tuple[bytes, bytes], int] = dict()
        for rank, merge in enumerate(self.merges):
            pair_ranks[merge] = rank
        return pair_ranks

    @classmethod
    def from_files(
        cls: Type["Tokenizer"],
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath) as vocab_f:
            vocab = json.load(vocab_f)

        merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(cleaned_line.split(" ")))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        # split into tests and special tokens with re.split
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        # 3. Pre-tokenization
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            split_pattern = f"({ '|'.join(map(re.escape, sorted_special_tokens)) })"
            splitted_text = re.split(split_pattern, text)

        else:
            splitted_text = [text]

        pre_tokens = []
        # first tqdm (len splitted texts)
        for sub_text in tqdm(splitted_text):
            if self.special_tokens and sub_text in self.special_tokens:
                pre_tokens.append((sub_text.encode("utf-8"), True))
            else:
                for pre_token in re.finditer(PAT, sub_text):
                    byte_representation = tuple(
                        bytes([b]) for b in pre_token.group().encode("utf-8")
                    )
                    pre_tokens.append((byte_representation, False))
        res = []
        # second tqdm (n pretokens)
        for pre_token, is_special in tqdm(pre_tokens):
            if is_special:
                res.append(tuple((pre_token,)))
            else:
                while True:
                    # print('--- iter ---')
                    min_pair_rank = float("inf")
                    min_pair_index = None
                    min_pair = None
                    for index, pair in enumerate(zip(pre_token, pre_token[1:])):
                        pair_rank = self.pair_ranks.get(pair, None)
                        if pair_rank is not None and pair_rank < min_pair_rank:
                            min_pair_rank = pair_rank
                            min_pair_index = index
                            min_pair = pair
                    if min_pair is None:
                        break
                    new_tuple_pairs = (
                        pre_token[:min_pair_index]
                        + (min_pair[0] + min_pair[1],)
                        + pre_token[min_pair_index + 2 :]
                    )
                    pre_token = tuple(new_tuple_pairs)
                res.append(pre_token)

        out = []
        # second tqdm (n pretokens -> indices)
        for r in tqdm(res):
            for token in r:
                out.append(self.inverse_vocab[token])
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8", errors="replace")
