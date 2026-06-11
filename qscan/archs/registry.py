"""Optional architecture registry.

The model registers itself here so it can be discovered by name. If you use a
framework that has its own registry (e.g. BasicSR), you can ignore this and
import :class:`QSCAN` directly.
"""

from __future__ import annotations

from typing import Callable, Dict, Type


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._obj_map: Dict[str, type] = {}

    def register(self, obj: Type = None):
        if obj is None:
            def deco(real_obj):
                self._do_register(real_obj.__name__, real_obj)
                return real_obj
            return deco
        self._do_register(obj.__name__, obj)
        return obj

    def _do_register(self, name: str, obj: type):
        self._obj_map[name] = obj

    def get(self, name: str):
        if name not in self._obj_map:
            raise KeyError(f"No object named '{name}' registered in '{self._name}'.")
        return self._obj_map[name]

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def keys(self):
        return self._obj_map.keys()


ARCH_REGISTRY = Registry('arch')

__all__ = ['ARCH_REGISTRY', 'Registry']
