from datetime import datetime
from pathlib import Path

import attrs

from ._validation import _NONNEGATIVE_INTEGER, _POSITIVE_INTEGER, _POSITIVE_NUMBER, _to_path
from .execution import LoggingConfig, RuntimeConfig
from .model import ModelConfig
from .tokenizer import TokenizerConfig


def _default_output_dir() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts/inference") / timestamp


@attrs.frozen(kw_only=True)
class EngineConfig:
    model: ModelConfig = attrs.field(
        validator=attrs.validators.instance_of(ModelConfig)
    )
    checkpoint: Path = attrs.field(converter=_to_path)
    runtime: RuntimeConfig = attrs.field(
        factory=RuntimeConfig, validator=attrs.validators.instance_of(RuntimeConfig)
    )


@attrs.frozen(kw_only=True)
class SchedulerConfig(EngineConfig):
    max_running_requests: int = attrs.field(
        default=1, validator=[_POSITIVE_INTEGER, attrs.validators.le(2**32 - 1)]
    )


@attrs.frozen(kw_only=True)
class LLMConfig(SchedulerConfig):
    tokenizer: TokenizerConfig = attrs.field(
        validator=attrs.validators.instance_of(TokenizerConfig)
    )


@attrs.frozen(kw_only=True)
class SamplingParams:
    """Per-request sampling settings and stopping conditions"""

    max_new_tokens: int = attrs.field(default=16, validator=_POSITIVE_INTEGER)
    temperature: float | None = attrs.field(
        default=None, validator=attrs.validators.optional(_POSITIVE_NUMBER)
    )
    top_p: float = attrs.field(
        default=1.0,
        validator=[
            attrs.validators.optional(_POSITIVE_NUMBER),
            attrs.validators.le(1.0),
        ]
    )
    top_k: int | None = attrs.field(
        default=None, validator=attrs.validators.optional(_POSITIVE_INTEGER)
    )
    eos_token_id: int | None = attrs.field(
        default=None, validator=attrs.validators.optional(_NONNEGATIVE_INTEGER)
    )


@attrs.frozen(kw_only=True)
class InferenceMapperConfig:
    llm: LLMConfig
    generation: SamplingParams = attrs.field(factory=SamplingParams)
    prompt: str | None = None
    input_path: Path | None = attrs.field(
        default=None, converter=attrs.converters.optional(_to_path)
    )
    output_dir: Path = attrs.field(factory=_default_output_dir, converter=_to_path)
    seed: int = attrs.field(
        default=42, validator=[_NONNEGATIVE_INTEGER, attrs.validators.le(2**32 - 1)]
    )
    logging: LoggingConfig = attrs.field(
        factory=LoggingConfig, validator=attrs.validators.instance_of(LoggingConfig)
    )


@attrs.frozen(kw_only=True)
class InferenceServerConfig:
    engine: EngineConfig
    generation: SamplingParams = attrs.field(factory=SamplingParams)
    host: str = "127.0.0.1"
    port: int = 8000
    max_pending_requests: int = 16
    seed: int = attrs.field(
        default=42, validator=[_NONNEGATIVE_INTEGER, attrs.validators.le(2**32 - 1)]
    )
    logging: LoggingConfig = attrs.field(
        factory=LoggingConfig, validator=attrs.validators.instance_of(LoggingConfig)
    )
