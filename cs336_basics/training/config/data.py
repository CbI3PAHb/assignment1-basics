"""Tokenized datasets and local batching configuration."""

from pathlib import Path
from typing import Literal, get_args

import attrs

from ._converters import _to_path
from ._validators import _POSITIVE_INTEGER

BatchingType = Literal["random_sequences"]
SUPPORTED_BATCHING_TYPES = get_args(BatchingType)

DatasetDType = Literal["uint16", "uint32"]
SUPPORTED_DATASET_DTYPES = get_args(DatasetDType)


@attrs.frozen(kw_only=True)
class BatchingConfig:
    """Random contiguous token sequences per local pass, before gradient accumulation."""

    sequence_length: int = attrs.field(validator=_POSITIVE_INTEGER)
    micro_batch_size: int = attrs.field(validator=_POSITIVE_INTEGER)
    type: BatchingType = attrs.field(
        default="random_sequences",
        validator=attrs.validators.in_(SUPPORTED_BATCHING_TYPES),
    )


@attrs.frozen(kw_only=True)
class DatasetConfig:
    """A flat, already tokenized binary file; dtype describes its stored token IDs"""

    path: Path = attrs.field(converter=_to_path)
    dtype: DatasetDType = attrs.field(
        default="uint16", validator=attrs.validators.in_(SUPPORTED_DATASET_DTYPES)
    )


@attrs.frozen(kw_only=True)
class DataConfig:
    train: DatasetConfig = attrs.field(
        validator=attrs.validators.instance_of(DatasetConfig)
    )
    eval: DatasetConfig = attrs.field(
        validator=attrs.validators.instance_of(DatasetConfig)
    )
    batching: BatchingConfig = attrs.field(
        validator=attrs.validators.instance_of(BatchingConfig)
    )
