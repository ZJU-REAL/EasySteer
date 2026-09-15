# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for :mod:`easysteer.extraction`."""

import warnings

from easysteer import extraction

warnings.warn(
    "easysteer.steer is deprecated; use easysteer.extraction instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = extraction.__all__


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(extraction, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
