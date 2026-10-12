"""Configure cattrs without coercing invalid scalar values from YAML."""

import cattrs


def _structure_integer(value: object, _type: type[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an integer, got {value!r} ({type(value).__name__})")
    return value


def _structure_float(value: object, _type: type[float]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected int or float, got {value!r} ({type(value).__name__})")
    return value


def _structure_boolean(value: object, _type: type[bool]) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"expected a boolean, got {value!r} ({type(value).__name__})")
    return value


def _make_converter() -> cattrs.Converter:
    converter = cattrs.Converter(
        forbid_extra_keys=True,
        prefer_attrib_converters=True,
        omit_if_default=False,
    )
    converter.register_structure_hook(int, _structure_integer)
    converter.register_structure_hook(float, _structure_float)
    converter.register_structure_hook(bool, _structure_boolean)
    return converter
