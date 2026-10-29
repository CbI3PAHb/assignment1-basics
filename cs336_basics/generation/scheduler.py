"""CPU scheduling contracts for one execution step at a time.

The driver calls schedule(), executes its plan once, then calls apply_result().
On execution failure it calls abort_all_requests() and re-raises the original error.
Only apply_result() advances computed-token counts and appends sampled tokens.

These docstrings specify the intended implementation; Scheduler is a learner
draft. A plan contains immutable CPU metadata, not GPU tensors.
"""

from collections import deque
from collections.abc import Sequence
from typing import Iterable, NamedTuple

import attrs

from cs336_basics.config import SamplingParams
from cs336_basics.config._validation import (
    _NONNEGATIVE_INTEGER,
    _POSITIVE_INTEGER,
)

from .kv_cache import BlockManager
from .requests import Req
from .results import FinishReason


def ceil(a: int, b: int) -> int:
    """Divide an integer count by a block size, rounding up.

    Args:
        a: Nonnegative item count, such as a token count.
        b: Positive number of items per block.

    Returns:
        int: Number of blocks needed; zero when a is zero.

    Raises:
        ValueError: If a < 0 or b <= 0

    Notes:
        Validate the count and block size before performing the division.
    """
    if a < 0:
        raise ValueError(f"got {a=} < 0")
    if b <= 0:
        raise ValueError(f"got {b=} < 0")

    return (a + b - 1) // b


@attrs.frozen(kw_only=True, eq=False)
class ScheduledRequest:
    """Immutable snapshot of one request's positive-length work in one step.

    Args:
        request_id: Required ORIGINAL Req.request_id. Scheduler supplies it;
            a work item never creates a new identity.
        input_token_ids: Nonempty tuple containing only the input tokens to compute
            in this step: req.token_ids[start_pos:start_pos + num_input_tokens].
            This is neither the complete history nor newly sampled output.
        start_pos: Number of tokens already computed before this step. The
            selected tokens have absolute positions start_pos, start_pos + 1,
            ..., start_pos + num_input_tokens - 1.
        block_table: Tuple of this request's physical block IDs, obtained after
            ensuring capacity for start_pos + num_input_tokens total tokens.
        sampling_params: Required explicit argument: the request's existing
            SamplingParams if this work item consumes its entire currently
            available input suffix; otherwise None. Intermediate prompt chunks
            use None and must not sample.

    Attributes:
        num_input_tokens: Read-only len(input_token_ids), derived rather than supplied.

    Raises:
        TypeError: If a required argument is missing or a field fails its attrs
            type validator. sampling_params=None is valid and does not raise.
        ValueError: If input_token_ids is empty or start_pos is negative.

    Notes:
        Fields cannot be reassigned after construction. This work item contains no
        mutable Req reference. Creating it does not advance request progress.
    """

    request_id: str = attrs.field(
        validator=attrs.validators.instance_of(str),
    )

    # ONLY this step's positive-length input slice.
    input_token_ids: tuple[int, ...] = attrs.field(
        validator=[
            attrs.validators.instance_of(tuple),
            attrs.validators.min_len(1),
        ]
    )

    # Number of tokens computed before this step.
    start_pos: int = attrs.field(validator=_NONNEGATIVE_INTEGER)

    # Snapshot after pages for this step are assigned
    block_table: tuple[int, ...] = attrs.field(validator=attrs.validators.instance_of(tuple))

    # Original immutable SamplingParams when this chunk reaches the end of available input; otherwise None means no sampling.
    sampling_params: SamplingParams | None = attrs.field(
        validator=attrs.validators.optional(attrs.validators.instance_of(SamplingParams))
    )

    @property
    def num_input_tokens(self) -> int:
        """Return the number of input tokens to compute for this request this step."""
        return len(self.input_token_ids)

    @property
    def do_sample(self) -> bool:
        """Whether this step should sample an output token for this request."""
        return self.sampling_params is not None


@attrs.frozen(kw_only=True)
class StepPlan:
    """Immutable description of all request work for one model forward.

    Args:
        step_id: Nonnegative execution ID assigned by this Scheduler. IDs of
            successive nonempty plans increase; callers do not choose them.
        requests: Nonempty tuple of ScheduledRequest snapshots, in execution order.
            Every item has positive num_input_tokens. Empty work is represented by
            schedule() returning None, not by an empty StepPlan.

    Attributes:
        sampling_request_ids: Derived tuple of request_id values for items whose
            sampling_params is not None, preserving execution order. It may be empty
            even when requests is nonempty. It must not be independent caller input.

    Raises:
        TypeError: If a field fails its attrs type validator.
        ValueError: If step_id is negative or requests is empty.

    Notes:
        Work items and IDs are snapshots, not references to mutable request state.
        The executor returns one sampled token per sampling_request_ids entry,
        not necessarily one per item. For requests A (decode) and B (partial prompt),
        sampling_request_ids is (A.request_id,) and the result has one token.
    """

    # Increasing execution ID and ordered nonempty work items.
    step_id: int = attrs.field(validator=_NONNEGATIVE_INTEGER)
    requests: tuple[ScheduledRequest, ...] = attrs.field(
        validator=[attrs.validators.instance_of(tuple), attrs.validators.min_len(1)]
    )

    @property
    def sampling_request_ids(self) -> tuple[str, ...]:
        """Return IDs of requests needing sampling, in execution order"""

        return tuple(req.request_id for req in self.requests if req.do_sample)


class PlanningResult(NamedTuple):
    scheduled_requests: list[ScheduledRequest]
    remaining_budget: int


@attrs.define(kw_only=True)
class Scheduler:
    """Own request admission, one-step planning, result commits and cleanup.

    Args:
        block_manager: Required existing CPU BlockManager used for reservations
            and physical block ownership. The driver supplies this instance;
            Scheduler does not create a default manager.
        max_step_input_tokens: Positive maximum sum of num_input_tokens across one plan.
        max_active_requests: Positive maximum number of active requests, including
            previously admitted requests omitted from a particular plan.
            Configure this at or below max_step_input_tokens. Running-first
            admission cannot reach a higher active count with a fixed token budget.
        max_prefill_chunk_tokens: Positive maximum prompt tokens per request per step.
            The remaining prompt length and step token budget also limit it.
        max_context_tokens: Positive maximum total history length per request,
            counting prompt and generated tokens together.

    Raises:
        TypeError: If a required argument is missing or a constructor field
            fails its attrs type validator.
        ValueError: If a numeric limit is nonpositive or boolean.
            Also if max_active_requests > max_step_input_tokens.

    Notes:
        Maintain known request identities, FIFO waiting/active order, and at
        most one outstanding plan. After schedule() returns a plan, apply or
        abort it before scheduling, submitting or cancelling anything else.
        Keep terminal identities known for duplicate rejection and idempotent
        cancellation. Callers must not mutate requests after submission.

        Select existing active work in admission order, then admit waiting
        requests only when their first chunk can enter this step. Req.is_prefill
        and Req.is_decode describe progress; they do not define separate queues
        or reorder active work. An ordered dict[str, Req] is a convenient active
        collection: iterate its values and remove a completed request by ID.

        The driver owns the generation loop. This class uses CPU metadata and
        never calls the model, samples tokens, or owns GPU tensor storage.
        The documented transitions and validations are the implementation task.
    """

    block_manager: BlockManager = attrs.field(
        validator=attrs.validators.instance_of(BlockManager),
    )
    max_step_input_tokens: int = attrs.field(validator=_POSITIVE_INTEGER, on_setattr=attrs.setters.frozen)
    max_active_requests: int = attrs.field(validator=_POSITIVE_INTEGER, on_setattr=attrs.setters.frozen)
    max_prefill_chunk_tokens: int = attrs.field(validator=_POSITIVE_INTEGER, on_setattr=attrs.setters.frozen)
    max_context_tokens: int = attrs.field(validator=_POSITIVE_INTEGER, on_setattr=attrs.setters.frozen)

    _requests_by_id: dict[str, Req] = attrs.field(
        init=False, factory=dict, validator=attrs.validators.instance_of(dict)
    )
    _waiting_request_ids: deque[str] = attrs.field(
        init=False, factory=deque, validator=attrs.validators.instance_of(deque)
    )
    _active_request_ids: list[str] = attrs.field(init=False, factory=list, validator=attrs.validators.instance_of(list))

    # TODO add associative container to store active requests -> we need to remove from the container with request's id
    # _active_requests: dict[str, Req] = attrs.field(init=False, factory=dict, validator=attrs.validators.instance_of(dict))
    _pending_plan: StepPlan | None = attrs.field(
        init=False, default=None, validator=attrs.validators.optional(attrs.validators.instance_of(StepPlan))
    )
    _next_step_id: int = attrs.field(init=False, default=0, validator=_NONNEGATIVE_INTEGER)

    def __attrs_post_init__(self):
        if self.max_active_requests > self.max_step_input_tokens:
            raise ValueError(f"{self.max_active_requests=}, {self.max_step_input_tokens=}")

    def _validate_req(self, *, req: Req) -> None:
        if req.request_id in self._requests_by_id:
            raise ValueError(f"{req.request_id=} is already known")

        if req.prompt_tokens_count == 0:
            raise ValueError("Prompt is empty")

        if req.prompt_tokens_count >= self.max_context_tokens:
            raise ValueError(f"Got {req.prompt_tokens_count=}, {self.max_context_tokens=}")

    def _validate_kv_cache_capacity(self, *, req: Req) -> None:
        max_possible_seq_len = min(
            self.max_context_tokens, req.prompt_tokens_count + req.sampling_params.max_new_tokens
        )
        needed_blocks: int = ceil(max_possible_seq_len, self.block_manager.block_size)
        # print(f"{max_possible_seq_len=}, {needed_blocks=}, {self.block_manager.num_blocks=}")
        if needed_blocks > self.block_manager.num_blocks:
            raise ValueError(f"the prompt too large, there is not enough KV cache blocks to fit its KVs")

    def add_request(self, *, req: Req) -> None:
        """Validate a new request and retain it for later FIFO admission.

        Args:
            req: Fresh request with its original prompt, stable request_id,
                sampling policy and zero computed-token count.

        Returns:
            None. An ordinary accepted request stays WAITING until schedule()
            admits it. A valid prompt already ending in configured EOS becomes
            FINISHED with FinishReason.EOS immediately, without any KV reserve.

        Raises:
            RuntimeError: If a plan is outstanding. Submission is allowed only
                between steps, after apply_result() or abort_all_requests().
            ValueError: If request_id is already known, the prompt is empty,
                or prompt_tokens_count >= max_context_tokens. For a nonterminal
                prompt, also raise if its lifetime block reservation cannot
                fit even in a completely empty physical pool.

        Notes:
            Reject invalid submissions before retaining the request or changing
            request state, reservations, counters or queues. Check identity and
            prompt validity first, then prompt EOS, then lifetime feasibility.
            A terminal EOS prompt does not need the feasibility check.

            Lifetime token capacity is min(max_context_tokens,
            prompt_tokens_count + sampling_params.max_new_tokens). Round it up
            by block_manager.block_size and compare against num_blocks.
            This is a TOTAL-pool check, not a check of available_block_count.
            Temporary credit shortage is normal: enqueue and wait, without
            raising. Actual credit reservation happens in schedule().
        """
        if self._pending_plan is not None:
            raise RuntimeError()  # kek, what is outstanding?

        self._validate_req(req=req)

        if req.sampling_params.eos_token_id == req.last_token:
            self._requests_by_id[req.request_id] = req
            req.finish(FinishReason.EOS)
            return

        self._validate_kv_cache_capacity(req=req)
        self._requests_by_id[req.request_id] = req
        self._waiting_request_ids.append(req.request_id)

    def _plan_request(self, *, req: Req, token_budget: int) -> ScheduledRequest | None:
        if token_budget < 0:
            raise ValueError(f"Negative token budget not allowed, got {token_budget=}")

        if token_budget == 0 or req.num_uncomputed_tokens == 0:
            return None

        num_input_tokens: int = min(
            req.num_uncomputed_tokens,
            token_budget,
            self.max_prefill_chunk_tokens,
        )

        if req.is_decode:
            assert num_input_tokens <= 1

        start_pos: int = req.num_computed_tokens
        end_pos: int = start_pos + num_input_tokens
        block_table: tuple[int, ...] = self.block_manager.ensure_capacity(
            request_id=req.request_id, num_tokens=req.num_computed_tokens + num_input_tokens
        )

        scheduled_request = ScheduledRequest(
            request_id=req.request_id,
            input_token_ids=tuple(req.token_ids[start_pos:end_pos]),
            start_pos=start_pos,
            block_table=block_table,
            sampling_params=req.sampling_params if end_pos == req.total_tokens_count else None,
        )
        return scheduled_request

    def _plan_active_requests(
        self,
        *,
        active_requests: Iterable[Req],
        token_budget: int,
    ) -> PlanningResult:
        if token_budget < 0:
            raise ValueError(f"Negative token budget not allowed, got {token_budget=}")

        scheduled_requests: list[ScheduledRequest] = []

        for req in active_requests:
            scheduled_request: ScheduledRequest | None = self._plan_request(req=req, token_budget=token_budget)

            if scheduled_request is not None:
                num_input_tokens = scheduled_request.num_input_tokens
                token_budget -= num_input_tokens
                scheduled_requests.append(scheduled_request)

            if token_budget <= 0:
                assert token_budget == 0, f"got {token_budget=}"
                break

        return PlanningResult(
            scheduled_requests=scheduled_requests,
            remaining_budget=token_budget,
        )

    def _plan_waiting_requests(self, *, token_budget: int) -> PlanningResult:
        scheduled_requests: list[ScheduledRequest] = []

        if token_budget < 0:
            raise ValueError(f"Negative token budget: {token_budget}")

        while (
            len(self._waiting_request_ids) > 0
            and len(self._active_request_ids) < self.max_active_requests
            and token_budget > 0
        ):
            request_id: str = self._waiting_request_ids[0]
            req: Req = self._requests_by_id[request_id]

            max_tokens: int = min(self.max_context_tokens, req.prompt_tokens_count + req.sampling_params.max_new_tokens)

            if not self.block_manager.reserve(request_id=req.request_id, max_tokens=max_tokens):
                break

            scheduled_request: ScheduledRequest | None = self._plan_request(req=req, token_budget=token_budget)
            if scheduled_request is None:
                raise RuntimeError("scheduled request must be not none")

            token_budget -= scheduled_request.num_input_tokens
            self._waiting_request_ids.popleft()
            self._active_request_ids.append(req.request_id)
            req.start()

            scheduled_requests.append(scheduled_request)

        return PlanningResult(
            scheduled_requests=scheduled_requests,
            remaining_budget=token_budget,
        )

    def schedule(self) -> StepPlan | None:
        """Prepare positive-length request work for exactly one model forward.

        First visit existing active requests in admission order, sharing one
        step input-token budget. Do not split or reorder this pass by phase.
        Select min(req.num_uncomputed_tokens, remaining_step_token_budget);
        for req.is_prefill, also cap the count by max_prefill_chunk_tokens.
        An ordinary decoder has one pending input token. A fresh one-token
        prompt or final one-token prompt chunk is still prefill. Omit zero work.

        Then consider FIFO waiting requests while token budget and active slots
        remain. Valid waiting requests have nonempty uncomputed prompts, so
        positive budget guarantees a first chunk. Reserve lifetime credits
        before calling _plan_request. On success, remove the waiting head, mark
        it RUNNING, append it to active membership and include its chunk in this
        plan. A failed reservation stops admission without bypassing that head.
        With no remaining token budget, leave waiting requests and their credits
        untouched. Already-active requests keep their existing reservations.

        For each chosen request, start_pos is its current computed count and
        input_token_ids is exactly the chosen uncomputed slice. Ensure physical KV
        capacity for start_pos + num_input_tokens and snapshot the returned table.
        Set sampling_params only when that slice reaches len(req.token_ids);
        otherwise use None. Freeze the ordered requests and store the pending plan.

        Returns:
            StepPlan | None: A nonempty immutable plan with an increasing
            step_id, or None when no work can be scheduled. Mixed decode and
            prefill requests, and plans with no sampling-eligible requests, are valid.

        Raises:
            RuntimeError: If the previous plan has not been applied or aborted.
                Reject the call before admission or allocation changes.

        Notes:
            This method may reserve credits, allocate pages and change WAITING
            to RUNNING. It must NOT advance num_computed_tokens, append output,
            or mark scheduled input as already executed. A valid reservation
            guarantees its request can grow even when no admission credits remain.
            Budget is spent on selected input tokens, never cached history or
            unused future output capacity. Active requests may be omitted when
            earlier work exhausts the budget; they retain their membership.
            An empty queue or temporary admission pressure is not an exception.
        """
        if self._pending_plan is not None:
            raise RuntimeError("the previous plan has not been applied or aborted")

        active_requests: list[Req] = [self._requests_by_id[req_id] for req_id in self._active_request_ids]

        plan_active: PlanningResult = self._plan_active_requests(
            active_requests=active_requests, token_budget=self.max_step_input_tokens
        )
        plan_waiting: PlanningResult = self._plan_waiting_requests(token_budget=plan_active.remaining_budget)

        requests_to_schedule = tuple(plan_active.scheduled_requests + plan_waiting.scheduled_requests)

        if len(requests_to_schedule) > 0:
            step_plan = StepPlan(
                step_id=self._next_step_id,
                requests=requests_to_schedule,
            )
            self._next_step_id += 1
            self._pending_plan = step_plan
            return step_plan
        return None

    def apply_result(self, *, plan: StepPlan, sampled_token_ids: Sequence[int]) -> None:
        """Commit one completed step and release requests that have finished.

        Args:
            plan: This Scheduler's current outstanding plan from schedule().
            sampled_token_ids: CPU token IDs in EXACT plan.sampling_request_ids
                order, one per eligible request. Pass an empty sequence for a
                plan consisting entirely of intermediate prompt chunks.

        Returns:
            None. Advance every planned request's num_computed_tokens by its
            corresponding item's num_input_tokens, and append one output token
            only to eligible requests.
            Clear the pending plan after a successful commit.

        Raises:
            RuntimeError: If there is no outstanding plan or the supplied plan
                is not the current one, including stale, applied or aborted plans.
            ValueError: If len(sampled_token_ids) != len(plan.sampling_request_ids).
                Validate plan and count BEFORE any request or ownership changes.
                On this error the pending plan remains usable with corrected
                results; do not advance counts or append even a partial result.

        Notes:
            A newly sampled token is appended but NOT counted as computed: it
            becomes input to a later step unless the request finishes now.
            After appending, check stopping in order: configured EOS, then
            max_new_tokens, then max_context_tokens. Keep a generated EOS token
            in history. Finish and release terminal requests' pages and credits;
            surviving requests stay active with their existing physical history.
            A short valid result sequence is normal when only some requests sample.
        """
        if self._pending_plan is None or self._pending_plan is not plan:
            raise RuntimeError("the previous plan has not been applied or aborted")

        if len(plan.sampling_request_ids) != len(sampled_token_ids):
            raise ValueError(f"{len(plan.sampling_request_ids)=}, {len(sampled_token_ids)=}")

        for request_id, sampled_token_id in zip(plan.sampling_request_ids, sampled_token_ids, strict=True):
            req: Req = self._requests_by_id[request_id]
            req.append_token(sampled_token_id)
            if sampled_token_id == req.sampling_params.eos_token_id:
                req.finish(FinishReason.EOS)

            elif req.generated_tokens_count >= req.sampling_params.max_new_tokens:
                assert req.generated_tokens_count == req.sampling_params.max_new_tokens
                req.finish(FinishReason.MAX_NEW_TOKENS)

            elif req.total_tokens_count >= self.max_context_tokens:
                assert req.total_tokens_count == self.max_context_tokens
                req.finish(FinishReason.MAX_CONTEXT_CAPACITY)

            if req.finished:
                self.block_manager.release(request_id=request_id)
                # del self._active_requests[request_id]
                self._active_request_ids.remove(request_id)

        for scheduled_request in plan.requests:
            req: Req = self._requests_by_id[scheduled_request.request_id]
            req.num_computed_tokens += scheduled_request.num_input_tokens

        self._pending_plan = None

    def cancel_request(self, *, request_id: str) -> None:
        """Cancel a known waiting or active request between execution steps.

        Args:
            request_id: ID previously accepted by this Scheduler.

        Returns:
            None. Finish a nonterminal request with FinishReason.CANCELLED,
            remove it from waiting/active work, and release any pages and credits.
            A known request already finished for any reason is an unchanged no-op.

        Raises:
            RuntimeError: If a plan is outstanding, before cancellation changes.
                This also applies when the target request is not in that plan.
            KeyError: If request_id is unknown to this Scheduler.

        Notes:
            Cancellation does not append tokens or advance computed counts.
            Waiting requests may have no reservation; that is not an error.
            Repeating cancellation between steps does not raise or free twice.
        """
        if self._pending_plan is not None:
            raise RuntimeError()  # TODO add runtimeerror message

        if request_id not in self._requests_by_id:
            raise KeyError()  # TODO add keyerror message

        req: Req = self._requests_by_id[request_id]
        if req.finished:
            return

        req.finish(FinishReason.CANCELLED)
        self.block_manager.release(request_id=request_id)

        # breakpoint()
        if self._active_request_ids.count(request_id):
            self._active_request_ids.remove(request_id)

        if request_id in self._waiting_request_ids:
            self._waiting_request_ids.remove(request_id)

    def _abort_active_request(self, *, request_id: str):
        assert self._active_request_ids.count(request_id) == 1
        assert request_id not in self._waiting_request_ids

        req: Req = self._requests_by_id[request_id]
        self._active_request_ids.remove(request_id)
        self.block_manager.release(request_id=request_id)
        req.abort()

    def _abort_waiting_request(self, *, request_id: str):
        assert self._active_request_ids.count(request_id) == 0
        assert request_id == self._waiting_request_ids[0]
        self._waiting_request_ids.popleft()

        req: Req = self._requests_by_id[request_id]
        self.block_manager.release(request_id=request_id)
        req.abort()

    def abort_all_requests(self, *, plan: StepPlan) -> None:
        """Abandon the outstanding step and fail all owned nonterminal requests.

        Args:
            plan: This Scheduler's current outstanding plan from schedule().

        Returns:
            None. Finish ALL owned nonterminal requests with FinishReason.ERROR,
            including waiting requests and active requests omitted from the plan.
            Release their pages and credits, clear waiting/active work, and clear
            the pending plan. Preserve previously terminal requests and reasons.

        Raises:
            RuntimeError: If there is no outstanding plan or plan is not the
                current one, including a repeated abort or an already applied
                plan. Reject before changing requests or ownership.

        Notes:
            Commit none of the failed step: leave every token history and
            computed-token count unchanged. Previously committed progress stays.
            This method performs cleanup; the driver then re-raises the original
            execution error and stops. Retrying partial GPU execution is outside
            this contract. Waiting requests without reservations are valid here.
        """
        if self._pending_plan is None:
            raise RuntimeError()  # TODO add runtimeerror message

        if plan is not self._pending_plan:
            raise RuntimeError()  # TODO add runtimeerror message

        self._pending_plan = None

        for scheduled_request in plan.requests:
            request_id = scheduled_request.request_id
            self._abort_active_request(request_id=request_id)

        while len(self._waiting_request_ids) > 0:
            waiting_request_id = self._waiting_request_ids[0]
            self._abort_waiting_request(request_id=waiting_request_id)
