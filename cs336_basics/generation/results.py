from enum import StrEnum, auto

import attrs


class FinishReason(StrEnum):
    EOS = auto()
    MAX_NEW_TOKENS = auto()
    MAX_CONTEXT_CAPACITY = auto()
    ERROR = auto()


@attrs.frozen(kw_only=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    finish_reason: FinishReason
    error: str | None = None
