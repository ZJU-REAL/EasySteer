# SPDX-License-Identifier: Apache-2.0
"""
Hidden States Capture for vLLM V1

A simple, clean interface for capturing hidden states from vLLM V1 models.
This module wraps vLLM V1's RPC-based hidden states capture to provide
a user-friendly API similar to V0.

Example:
    >>> import easysteer.hidden_states_v1 as hs
    >>> from vllm import LLM
    >>> 
    >>> llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    >>> all_hidden_states, outputs = hs.get_all_hidden_states(llm, ["Hello world"])
    >>> print(f"Captured {len(all_hidden_states)} samples")
"""

from .capture import get_all_hidden_states, HiddenStatesCaptureV1

__all__ = [
    "get_all_hidden_states",
    "HiddenStatesCaptureV1",
]

__version__ = "1.0.0"

