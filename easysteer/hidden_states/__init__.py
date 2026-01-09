# SPDX-License-Identifier: Apache-2.0
"""
Hidden States Capture for vLLM V1

A simple, clean interface for capturing hidden states from vLLM V1 models.
This module wraps vLLM V1's RPC-based hidden states capture to provide
a user-friendly API similar to V0.

Example:
    >>> import easysteer.hidden_states as hs
    >>> from vllm import LLM
    >>> 
    >>> # Capture hidden states (embed task)
    >>> llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", task="embed")
    >>> all_hidden_states, outputs = hs.get_all_hidden_states(llm, ["Hello world"])
    >>> print(f"Captured {len(all_hidden_states)} samples")
    >>> 
    >>> # Capture hidden states (generate task, for multimodal models)
    >>> llm_vlm = LLM(model="Qwen2.5-VL-7B-Instruct")
    >>> hidden_states, outputs = hs.get_all_hidden_states_generate(llm_vlm, ["Hello"])
    >>> print(f"Captured {len(hidden_states)} samples")
    >>> 
    >>> # Capture MoE router logits
    >>> llm_moe = LLM(model="mistralai/Mixtral-8x7B-v0.1")
    >>> router_logits, outputs = hs.get_moe_router_logits(llm_moe, ["Hello world"])
    >>> print(f"Captured {len(router_logits)} MoE layers")
"""

from .capture import get_all_hidden_states, HiddenStatesCaptureV1
from .capture_generate import (
    get_all_hidden_states_generate,
    HiddenStatesCaptureGenerate,
)
from .moe_capture import (
    get_moe_router_logits,
    analyze_expert_usage,
    MoERouterLogitsCaptureV1,
)
from .moe_capture_generate import (
    get_moe_router_logits_generate,
    MoERouterLogitsCaptureGenerate,
)

__all__ = [
    # Hidden states (embed task)
    "get_all_hidden_states",
    "HiddenStatesCaptureV1",
    # Hidden states (generate task)
    "get_all_hidden_states_generate",
    "HiddenStatesCaptureGenerate",
    # MoE router logits (embed task)
    "get_moe_router_logits",
    "analyze_expert_usage",
    "MoERouterLogitsCaptureV1",
    # MoE router logits (generate task)
    "get_moe_router_logits_generate",
    "MoERouterLogitsCaptureGenerate",
]

__version__ = "1.0.0"

