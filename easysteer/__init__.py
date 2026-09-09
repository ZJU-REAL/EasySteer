"""EasySteer: high-performance LLM steering."""

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["hidden_states", "steer", "reft"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
