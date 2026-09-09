# SPDX-License-Identifier: Apache-2.0
"""MoE router-logits capture with per-sample or concatenated results."""

from typing import Any

import torch

from .capture_result import capture


def get_moe_router_logits_generate(
    llm: Any,
    prompts: list[str] | list[dict[str, Any]],
    max_tokens: int = 1,
    split_by_samples: bool = False,
    **generate_kwargs,
) -> tuple[dict[int, torch.Tensor], Any] | tuple[list[dict[int, torch.Tensor]], Any]:
    """Capture MoE router logits while running generate.

    Works for any generate-capable MoE model, including multimodal
    ones (e.g. Qwen3-VL). With the default ``max_tokens=1`` only the
    prompt forward is captured. When router-logits steering is active,
    the captured logits are the post-steering ones.

    Args:
        llm: Single-worker vLLM LLM instance (compiled or eager,
            prefix caching on or off).
        prompts: text prompts, or multimodal dicts with ``prompt`` and
            ``multi_modal_data`` keys.
        max_tokens: tokens to generate (1 = prompt-only forward).
        split_by_samples: if True return one ``{layer_id: tensor}``
            dict per sample; if False return a single dict with all
            samples' rows concatenated per layer.
        **generate_kwargs (Any): forwarded into SamplingParams.

    Returns:
        ``(router_logits, outputs)`` where router_logits is
        ``{layer_id: (rows, n_experts)}`` (concatenated) or
        ``[sample_idx]{layer_id: (rows, n_experts)}`` (split).
    """
    result = capture(
        llm,
        prompts,
        max_tokens=max_tokens,
        stream="router_logits",
        **generate_kwargs,
    )
    if split_by_samples:
        return [result.sample(i) for i in range(len(result))], result.outputs
    return dict(result.layers), result.outputs
