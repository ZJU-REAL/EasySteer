# SPDX-License-Identifier: Apache-2.0
"""MoE router-logits capture through the generate task.

Compatibility wrapper: keeps the ``get_moe_router_logits_generate()``
signature while delegating to :func:`easysteer.hidden_states.capture`
on the ``router_logits`` stream, with exact label-driven per-sample
splitting.
"""

from typing import Any, Dict, List, Tuple, Union

import torch

from .capture_result import capture


def get_moe_router_logits_generate(
    llm: Any,
    prompts: Union[List[str], List[Dict[str, Any]]],
    max_tokens: int = 1,
    split_by_samples: bool = False,
    **generate_kwargs,
) -> Union[
    Tuple[Dict[int, torch.Tensor], Any],
    Tuple[List[Dict[int, torch.Tensor]], Any],
]:
    """Capture MoE router logits while running generate.

    Works for any generate-capable MoE model, including multimodal
    ones (e.g. Qwen3-VL). With the default ``max_tokens=1`` only the
    prompt forward is captured. When router-logits steering is active,
    the captured logits are the post-steering ones.

    Args:
        llm: vLLM LLM instance (any engine config: compiled or eager,
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
