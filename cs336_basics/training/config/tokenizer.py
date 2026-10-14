from pathlib import Path

import attrs

from ._converters import _to_path


@attrs.frozen(kw_only=True)
class TokenizerConfig:
    path: Path = attrs.field(converter=_to_path)
