from cs336_basics.config.tokenizer import TokenizerConfig

from .tokenizer import Tokenizer


def build_tokenizer(config: TokenizerConfig) -> Tokenizer:
    return Tokenizer.from_pickle(config.path)
