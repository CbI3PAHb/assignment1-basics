"""Engine.execute(plan=StepPlan, kv_store=PagedKVStore) packs one model call.

Positive query slices follow plan order; absolute positions and physical slots
follow each request's history/table. Only rows with sampling_params contribute
last-query logits and policies. ModelOutput.next_tokens_gpu is device int64[S]
in plan.sampling_request_ids order; intermediate chunks run without sampling.
The caller owns scheduling, page assignment and committing sampled results.
"""

import importlib

import pytest
import torch
from torch import nn

from cs336_basics.config import EngineConfig, ModelConfig, RuntimeConfig, SamplingParams
from cs336_basics.generation.results import ModelOutput
from cs336_basics.generation.sampling import Sampler
from cs336_basics.generation.scheduler import ScheduledRequest, StepPlan
from cs336_basics.transformers.paged_attention import PagedForwardContext
from cs336_basics.transformers.paged_kv import PagedKVStore
from cs336_basics.transformers.transformers import TransformerLM

_MODEL = dict(
    vocab_size=32,
    context_length=16,
    d_model=12,
    num_layers=2,
    num_heads=3,
    d_ff=24,
    rope_theta=10_000.0,
)


@pytest.fixture
def make_engine(monkeypatch, tmp_path, execution_device):
    """Keep real construction/loading boundaries, without checkpoint artifacts."""
    module = importlib.import_module("cs336_basics.generation.engine")

    def build(model):
        config = EngineConfig(
            model=ModelConfig(**_MODEL),
            checkpoint=tmp_path / "unused.pt",
            runtime=RuntimeConfig(device=execution_device.type),
        )
        monkeypatch.setattr(module, "build_model", lambda config: model)

        def load(*, src, model):
            assert src == config.checkpoint

        monkeypatch.setattr(module, "load_model_weights", load)
        engine = module.Engine(config)
        if not callable(getattr(engine, "execute", None)):
            pytest.fail("Implement Engine.execute(plan=..., kv_store=...)", pytrace=False)
        assert engine.model is model
        assert isinstance(engine.sampler, Sampler)
        assert not model.training
        return engine

    return build


def _store(device):
    store = PagedKVStore(
        num_layers=2,
        num_blocks=10,
        block_size=2,
        num_heads=3,
        head_dim=4,
        device=device,
        dtype=torch.float32,
    )
    store.k_pages.fill_(float("nan"))
    store.v_pages.fill_(float("nan"))
    return store


def _row(name, tokens, start, table, policy):
    return ScheduledRequest(
        request_id=name,
        input_token_ids=tuple(tokens),
        start_pos=start,
        block_table=tuple(table),
        sampling_params=policy,
    )


def _assert_int_tensor(actual, expected, device):
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype == torch.int64
    assert actual.device == device
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.int64, device=device))


def _assert_output(output, expected, device):
    assert isinstance(output, ModelOutput)
    _assert_int_tensor(output.next_tokens_gpu, expected, device)


class PackedLM(nn.Module):
    def __init__(self, forward_body):
        super().__init__()
        self.register_buffer("anchor", torch.empty(0), persistent=False)
        self.forward_body = forward_body
        self.calls = 0

    def forward(self, token_ids, *, token_positions, paged_context):
        assert not self.training
        assert torch.is_inference_mode_enabled(), "Model/page writes need inference mode"
        assert token_ids.ndim == 1, "Packed inputs have no padding/batch dimension"
        assert token_ids.dtype == token_positions.dtype == torch.int64
        assert token_ids.shape == token_positions.shape
        assert token_ids.device == token_positions.device == self.anchor.device
        assert isinstance(paged_context, PagedForwardContext)
        self.calls += 1
        return self.forward_body(token_ids, token_positions, paged_context)


def test_execute_packs_absolute_positions_pages_and_only_eligible_last_logits(
    make_engine,
    monkeypatch,
    execution_device,
):
    device = execution_device
    store = _store(device)
    policy = SamplingParams()
    plan = StepPlan(
        step_id=7,
        requests=(
            _row("partial-A", (11, 12), 3, (6, 1, 8), None),
            _row("B", (21,), 0, (3,), policy),
            _row("C", (4, 5), 1, (9, 0), policy),
        ),
    )
    before = repr(plan)
    logits = torch.arange(5 * 32, dtype=torch.float32, device=device).reshape(5, 32)

    def forward(ids, positions, context):
        assert context.kv_store is store
        _assert_int_tensor(ids, [11, 12, 21, 4, 5], device)
        _assert_int_tensor(positions, [3, 4, 0, 1, 2], device)
        metadata = context.metadata
        _assert_int_tensor(metadata.q_offsets, [0, 2, 3, 5], device)
        _assert_int_tensor(metadata.kv_lengths, [5, 1, 3], device)
        _assert_int_tensor(metadata.block_tables, [[6, 1, 8], [3, -1, -1], [9, 0, -1]], device)
        _assert_int_tensor(metadata.write_slots, [3, 16, 6, 19, 0], device)
        assert metadata.q_offsets_cpu == (0, 2, 3, 5)
        assert metadata.kv_lengths_cpu == (5, 1, 3)
        return logits.clone()

    sampled = []

    def sample(self, last_token_logits, sampling_params):
        assert torch.is_inference_mode_enabled(), "Sampling also needs inference mode"
        assert sampling_params.is_greedy
        torch.testing.assert_close(last_token_logits, logits[[2, 4]])
        sampled.append(True)
        return torch.tensor([5, 7], dtype=torch.int64, device=device)

    monkeypatch.setattr(Sampler, "sample", sample)
    model = PackedLM(forward)
    output = make_engine(model).execute(plan=plan, kv_store=store)
    _assert_output(output, [5, 7], device)
    assert plan.sampling_request_ids == ("B", "C")
    assert repr(plan) == before
    assert model.calls == 1 and sampled == [True]


def test_intermediate_only_execute_runs_model_without_sampling_or_rng(
    make_engine,
    monkeypatch,
    execution_device,
):
    device = execution_device
    model = PackedLM(lambda ids, positions, context: torch.zeros((len(ids), 32), device=device))
    engine = make_engine(model)
    store = _store(device)
    plan = StepPlan(
        step_id=0,
        requests=(
            _row("A", (11, 24), 0, (6,), None),
            _row("B", (4,), 0, (3,), None),
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Intermediate-only execution must not sample")

    monkeypatch.setattr(Sampler, "sample", forbidden)
    monkeypatch.setattr(torch, "multinomial", forbidden)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    output = engine.execute(plan=plan, kv_store=store)
    _assert_output(output, [], device)
    assert model.calls == 1
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)


class RecordingLM(TransformerLM):
    def __init__(self):
        super().__init__(**_MODEL)
        self.calls = []

    def forward(self, token_ids, *, token_positions, paged_context):
        assert not self.training and torch.is_inference_mode_enabled()
        assert token_ids.ndim == 1
        self.calls.append(token_ids.numel())
        return super().forward(
            token_ids,
            token_positions=token_positions,
            paged_context=paged_context,
        )


def test_real_model_mixed_chunked_and_reordered_execution_matches_full_prefixes(
    make_engine,
    monkeypatch,
    execution_device,
):
    device = execution_device
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(60336)
        model, reference = RecordingLM(), TransformerLM(**_MODEL)
    reference.load_state_dict(model.state_dict())
    reference.to(device).eval()
    engine = make_engine(model)
    store = _store(device)
    pointers = store.k_pages.data_ptr(), store.v_pages.data_ptr()
    tables = {"A": (6, 1, 8), "B": (3, 9, 0)}
    histories = {"A": [], "B": []}
    policy = SamplingParams()
    expected_logits, sampled, calls = [], [], []
    original_sample = Sampler.sample

    def sample(self, last_token_logits, sampling_params):
        assert torch.is_inference_mode_enabled() and sampling_params.is_greedy
        expected = torch.stack(expected_logits)
        assert torch.isfinite(last_token_logits).all()
        torch.testing.assert_close(last_token_logits, expected, rtol=1e-5, atol=1e-6)
        sampled.append(True)
        return original_sample(self, last_token_logits, sampling_params)

    monkeypatch.setattr(Sampler, "sample", sample)

    def run(entries):
        expected_logits.clear()
        rows = []
        with torch.inference_mode():
            for name, tokens, eligible in entries:
                start = len(histories[name])
                histories[name].extend(tokens)
                table = tables[name][: (len(histories[name]) + 1) // 2]
                rows.append(_row(name, tokens, start, table, policy if eligible else None))
                if eligible:
                    ids = torch.tensor([histories[name]], dtype=torch.int64, device=device)
                    expected_logits.append(reference(ids)[0, -1])
        plan = StepPlan(step_id=len(calls), requests=tuple(rows))
        before = repr(plan)
        output = engine.execute(plan=plan, kv_store=store)
        expected_tokens = [int(logits.argmax().item()) for logits in expected_logits]
        _assert_output(output, expected_tokens, device)
        calls.append(sum(len(tokens) for _, tokens, _ in entries))
        assert model.calls == calls, "Each plan needs exactly one packed model forward"
        assert len(sampled) == len(calls)
        assert (store.k_pages.data_ptr(), store.v_pages.data_ptr()) == pointers
        assert repr(plan) == before
        return dict(zip(plan.sampling_request_ids, expected_tokens, strict=True))

    first = run((("A", [11, 24, 7], True),))
    second = run((("A", [first["A"]], True), ("B", [4, 8], False)))
    third = run((("B", [6, 10, 12], True), ("A", [second["A"]], True)))
    run((("B", [third["B"]], True),))


def test_sparse_eligibility_preserves_mixed_original_sampling_policies(
    make_engine,
    monkeypatch,
    execution_device,
):
    device = execution_device
    greedy = SamplingParams()
    topk = SamplingParams(temperature=0.5, top_k=3)
    nucleus = SamplingParams(temperature=1.5, top_p=0.7)
    plan = StepPlan(
        step_id=4,
        requests=(
            _row("partial-A", (11, 12), 0, (6,), None),
            _row("greedy", (4,), 0, (3,), greedy),
            _row("partial-C", (8,), 0, (9,), None),
            _row("topk", (7, 6), 0, (1,), topk),
            _row("nucleus", (5,), 0, (8,), nucleus),
        ),
    )
    logits = torch.arange(7 * 32, dtype=torch.float32, device=device).reshape(7, 32)
    model = PackedLM(lambda ids, positions, context: logits.clone())
    observed = []

    def sample(self, last_token_logits, sampling_params):
        assert torch.is_inference_mode_enabled()
        torch.testing.assert_close(last_token_logits, logits[[2, 5, 6]])
        temperatures = sampling_params.temperatures
        assert temperatures.shape == (3, 1) and temperatures.device == device
        # The greedy mask owns row 0; its numeric temperature is unspecified.
        torch.testing.assert_close(
            temperatures[1:],
            torch.tensor([[0.5], [1.5]], device=device),
            check_dtype=False,
        )
        for name, values in (
            ("top_k", [[32], [3], [32]]),
            ("top_p", [[1.0], [1.0], [0.7]]),
            ("greedy_mask", [True, False, False]),
        ):
            actual = getattr(sampling_params, name)
            assert actual.device == device
            expected = torch.tensor(values, device=device)
            torch.testing.assert_close(actual, expected, check_dtype=False)
        observed.append(True)
        return torch.tensor([5, 9, 7], dtype=torch.int64, device=device)

    monkeypatch.setattr(Sampler, "sample", sample)
    before = repr(plan)
    output = make_engine(model).execute(plan=plan, kv_store=_store(device))
    _assert_output(output, [5, 9, 7], device)
    assert plan.sampling_request_ids == ("greedy", "topk", "nucleus")
    assert repr(plan) == before
    assert model.calls == 1 and observed == [True]


def test_model_failure_propagates_original_exception_without_sampling(
    make_engine,
    monkeypatch,
    execution_device,
):
    failure = RuntimeError("model failed after entering packed forward")

    def fail(ids, positions, context):
        raise failure

    def forbidden(*args, **kwargs):
        pytest.fail("Do not sample after a failed model call")

    model = PackedLM(fail)
    engine = make_engine(model)
    plan = StepPlan(step_id=0, requests=(_row("A", (11, 24), 0, (6,), SamplingParams()),))
    before = repr(plan)
    monkeypatch.setattr(Sampler, "sample", forbidden)
    with pytest.raises(RuntimeError) as caught:
        engine.execute(plan=plan, kv_store=_store(execution_device))
    assert caught.value is failure
    assert model.calls == 1 and repr(plan) == before
