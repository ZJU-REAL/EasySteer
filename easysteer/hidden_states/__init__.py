# SPDX-License-Identifier: Apache-2.0
"""Hidden-state and router-logit capture clients for vllm-steer.

The primary entry point is :func:`capture`, which returns a
:class:`CaptureResult` with exact, label-driven per-sample views.
``get_all_hidden_states_generate`` and ``get_moe_router_logits_generate``
are thin compatibility wrappers over it, keeping the signatures the
replication notebooks use.

Example:
    >>> import easysteer.hidden_states as hs
    >>> from vllm import LLM
    >>>
    >>> llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
    >>> result = hs.capture(llm, ["Hello world"])
    >>> result.sample(0)[10].shape  # sample 0, layer 10
"""

from .capture_result import CaptureResult, capture
from .capture_generate import get_all_hidden_states_generate
from .moe_capture_generate import get_moe_router_logits_generate

__all__ = [
    "CaptureResult",
    "capture",
    "get_all_hidden_states_generate",
    "get_moe_router_logits_generate",
]

__version__ = "2.0.0"
