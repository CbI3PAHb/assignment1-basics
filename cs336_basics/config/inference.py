from datetime import datetime
from pathlib import Path

import attrs

from ._converters import _to_path
from ._validators import _NONNEGATIVE_INTEGER
from .logging import LoggingConfig
from .model import ModelConfig
from .runtime import RuntimeConfig
from .tokenizer import TokenizerConfig


def _default_output_dir() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts/inference") / timestamp


@attrs.frozen(kw_only=True)
class InferenceEngineConfig:
    model: ModelConfig = attrs.field(
        validator=attrs.validators.instance_of(ModelConfig)
    )
    checkpoint: Path = attrs.field(converter=_to_path)
    tokenizer: TokenizerConfig = attrs.field(
        validator=attrs.validators.instance_of(TokenizerConfig)
    )
    runtime: RuntimeConfig = attrs.field(
        factory=RuntimeConfig, validator=attrs.validators.instance_of(RuntimeConfig)
    )


@attrs.frozen(kw_only=True)
class GenerationConfig:
    """Sampling settings"""

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0


@attrs.frozen(kw_only=True)
class InferenceMapperConfig:
    engine: InferenceEngineConfig
    generation: GenerationConfig = attrs.field(factory=GenerationConfig)
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
    engine: InferenceEngineConfig
    generation: GenerationConfig = attrs.field(factory=GenerationConfig)
    host: str = "127.0.0.1"
    port: int = 8000
    max_pending_requests: int = 16
    seed: int = attrs.field(
        default=42, validator=[_NONNEGATIVE_INTEGER, attrs.validators.le(2**32 - 1)]
    )
    logging: LoggingConfig = attrs.field(
        factory=LoggingConfig, validator=attrs.validators.instance_of(LoggingConfig)
    )
