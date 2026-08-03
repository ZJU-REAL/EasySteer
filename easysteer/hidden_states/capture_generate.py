# SPDX-License-Identifier: Apache-2.0
"""Hidden-state capture through the generate task.

Compatibility wrapper: keeps the long-standing
``get_all_hidden_states_generate()`` signature (used by the replication
notebooks and the frontend) while delegating capture, transport and
exact per-sample splitting to :func:`easysteer.hidden_states.capture`.
Splitting is always label-driven — the engine tags every captured row
with its owning request, which is the only correct attribution under
continuous batching.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .capture_result import capture


def _sugar_select(
    token_ids: Optional[List[int]],
    positions: Optional[List[int]],
    select: Optional[Any],
) -> Optional[Any]:
    """Translate the token_ids/positions sugar into a SelectSpec.

    Both filters select the union of their matches over prompt and
    generated tokens, matching the engine-side clause semantics.
    """
    if select is not None and (token_ids is not None or positions is not None):
        raise ValueError(
            "select cannot combine with the token_ids/positions "
            "shortcuts; put the filters inside the select clause"
        )
    if token_ids is None and positions is None:
        return select
    if token_ids is not None and not token_ids:
        raise ValueError("token_ids must be None or non-empty")
    if positions is not None and not positions:
        raise ValueError(
            "positions must be None or non-empty ('all rows' is the "
            "default when no filter is given)"
        )
    from vllm.steer_vectors.api import SelectSpec

    return SelectSpec(
        phases=["prompt", "generation"],
        tokens=list(token_ids) if token_ids is not None else None,
        positions=list(positions) if positions is not None else None,
    )


def get_all_hidden_states_generate(
    llm: Any,
    prompts: Union[List[str], List[Dict[str, Any]]],
    max_tokens: int = 1,
    split_by_samples: bool = True,
    token_ids: Optional[List[int]] = None,
    positions: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    dtype: Optional[str] = None,
    select: Optional[Union[dict, Any]] = None,
    **generate_kwargs,
) -> Union[
    Tuple[List[List[torch.Tensor]], Any], Tuple[List[torch.Tensor], Any]
]:
    """Capture every layer's hidden states while running generate.

    Works for any generate-capable model, including multimodal models
    (Qwen-VL, LLaVA, ...) that do not support the embed task. With the
    default ``max_tokens=1`` only the prompt forward is captured,
    matching what an embed task would produce.

    Args:
        llm: vLLM LLM instance (enforce_eager, prefix caching off).
        prompts: text prompts, or multimodal dicts with ``prompt`` and
            ``multi_modal_data`` keys.
        max_tokens: tokens to generate (1 = prompt-only forward).
        split_by_samples: if True return ``[sample][layer]`` tensors;
            if False return per-layer tensors concatenated over samples.
        token_ids: only capture rows whose input token id is in this
            list (source-side filter; unions with ``positions``).
        positions: only capture these absolute positions (negatives
            resolve from the prompt end; unions with ``token_ids``).
        layers: layer-id subset (None = all hooked layers).
        dtype: engine-side storage dtype (e.g. ``'float16'``).
        select: SelectSpec (or wire dict) — the full where-clause
            selection language; cannot combine with the shortcuts.
        **generate_kwargs (Any): forwarded into SamplingParams.

    Returns:
        ``(hidden_states, outputs)`` where hidden_states is
        ``[sample][layer]`` (split) or ``[layer]`` (concatenated),
        layers ordered by layer id.
    """
    result = capture(
        llm,
        prompts,
        max_tokens=max_tokens,
        layers=layers,
        dtype=dtype,
        select=_sugar_select(token_ids, positions, select),
        **generate_kwargs,
    )
    if split_by_samples:
        return result.to_nested(), result.outputs
    return [result.layers[lid] for lid in result.layer_ids], result.outputs
