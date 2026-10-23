from enum import StrEnum, auto
from uuid import uuid4

import attrs

from cs336_basics.config import SamplingParams
from cs336_basics.config._validation import _TOKEN_LIST

from .results import FinishReason


class ReqStatus(StrEnum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@attrs.define(kw_only=True)
class Req:
    """Persistent request state: identity, tokens, sampling policy and progress"""

    token_ids: list[int] = attrs.field(validator=_TOKEN_LIST)
    _id: str = attrs.field(
        factory=lambda: uuid4().hex,
        validator=attrs.validators.instance_of(str),
    )
    prompt_tokens_count: int = attrs.field(init=False)
    status: ReqStatus = ReqStatus.WAITING
    finish_reason: FinishReason | None = None
    sampling_params: SamplingParams = attrs.field(
        factory=SamplingParams,
        validator=attrs.validators.instance_of(SamplingParams),
    )

    def __attrs_post_init__(self) -> None:
        self.prompt_tokens_count = len(self.token_ids)

    @property
    def total_tokens_count(self) -> int:
        return len(self.token_ids)

    @property
    def generated_tokens_count(self) -> int:
        return len(self.token_ids) - self.prompt_tokens_count

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def finished(self) -> bool:
        return self.status == ReqStatus.FINISHED

    def finish(self, reason: FinishReason) -> None:
        self.status = ReqStatus.FINISHED
        self.finish_reason = reason

    def start(self) -> None:
        self.status = ReqStatus.RUNNING


@attrs.define(kw_only=True)
class Batch:
    """Requests for one execution step; per-step input metadata belongs here"""

    reqs: list[Req]
