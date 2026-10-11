"""Read YAML configuration files without silently overwriting duplicate keys."""

from collections.abc import Hashable
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError
from yaml.error import Mark
from yaml.nodes import MappingNode


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[object, object]:
        # Check the resulting mapping, including keys brought in by YAML merges.
        self.flatten_mapping(node)
        seen: dict[Hashable, Mark] = {}
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                )
            if key in seen:
                raise ConstructorError(
                    "first occurrence of this key",
                    seen[key],
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen[key] = key_node.start_mark
        return super().construct_mapping(node, deep=deep)
