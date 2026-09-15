# SPDX-License-Identifier: Apache-2.0
"""EasySteer: capture, extract, and train vectors for vLLM steering."""

from importlib import import_module
from types import ModuleType

__version__ = "0.29.0"
__all__ = ["capture", "extraction", "training", "vectors"]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__ and name not in ("hidden_states", "steer"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
