# SPDX-License-Identifier: Apache-2.0
"""Labelled activation capture for vllm-steer components.

The primary entry point is ``capture``, which returns a
``CaptureResult`` with labelled per-sample views.
``capture_batches`` yields batches for bounded processing or storage.

Example:
    >>> import easysteer.hidden_states as hs
    >>> from vllm import LLM
    >>>
    >>> llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
    >>> result = hs.capture(llm, ["Hello world"])
    >>> result.sample(0)[10].shape  # sample 0, layer 10
"""

from .capture_result import CaptureResult, capture, capture_batches

__all__ = [
    "CaptureResult",
    "capture",
    "capture_batches",
]
