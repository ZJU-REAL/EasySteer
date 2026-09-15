# SPDX-License-Identifier: Apache-2.0
"""Capture hidden states, attention head outputs, and MoE router logits.

``capture`` returns a ``CaptureResult`` with labelled per-sample views.
``capture_batches`` yields batches for bounded processing or storage.
"""

from .api import capture, capture_batches
from .result import CaptureResult

__all__ = ["CaptureResult", "capture", "capture_batches"]
