# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for :mod:`easysteer.capture`."""

import warnings

from easysteer.capture import CaptureResult, capture, capture_batches

warnings.warn(
    "easysteer.hidden_states is deprecated; use easysteer.capture instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CaptureResult", "capture", "capture_batches"]
