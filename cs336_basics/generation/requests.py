from enum import StrEnum, auto
from uuid import uuid4

import attrs

from cs336_basics.config import SamplingParams
from cs336_basics.config._validation import _NONNEGATIVE_INTEGER, _TOKEN_LIST

from .results import FinishReason


class ReqStatus(StrEnum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@attrs.define(kw_only=True)
class Req:
    """Persistent request state: identity, tokens, sampling policy and progress.

    Read-only phase properties required by the scheduling task:
        is_prefill: num_computed_tokens < prompt_tokens_count.
        is_decode: num_computed_tokens >= prompt_tokens_count.

    Derive both properties from progress; do not store another mutable phase.
    A final one-token prompt chunk is still prefill until its result is applied.
    These describe progress only, not eligibility: terminal requests must already
    be excluded from active work. Their implementation remains part of the task.
    """

    token_ids: list[int] = attrs.field(validator=_TOKEN_LIST)
    _id: str = attrs.field(
        factory=lambda: uuid4().hex,
        validator=attrs.validators.instance_of(str),
        on_setattr=attrs.setters.frozen,
    )
    status: ReqStatus = ReqStatus.WAITING
    finish_reason: FinishReason | None = None
    sampling_params: SamplingParams = attrs.field(
        factory=SamplingParams,
        validator=attrs.validators.instance_of(SamplingParams),
    )
    num_computed_tokens: int = attrs.field(
        default=0,
        init=False,
        validator=_NONNEGATIVE_INTEGER,
    )

    prompt_tokens_count: int = attrs.field(init=False, on_setattr=attrs.setters.frozen)

    @prompt_tokens_count.default
    def _initial_prompt_tokens_count(self) -> int:
        return len(self.token_ids)

    @property
    def total_tokens_count(self) -> int:
        return len(self.token_ids)

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.prompt_tokens_count

    @property
    def is_decode(self) -> bool:
        # return self.num_computed_tokens >= self.prompt_tokens_count
        return not self.is_prefill

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.total_tokens_count - self.num_computed_tokens

    @property
    def generated_tokens_count(self) -> int:
        return self.total_tokens_count - self.prompt_tokens_count

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def finished(self) -> bool:
        return self.status == ReqStatus.FINISHED

    @property
    def request_id(self) -> str:
        return self._id

    def finish(self, reason: FinishReason) -> None:
        self.status = ReqStatus.FINISHED
        self.finish_reason = reason

    def abort(self) -> None:
        self.finish(FinishReason.ERROR)

    def start(self) -> None:
        self.status = ReqStatus.RUNNING
