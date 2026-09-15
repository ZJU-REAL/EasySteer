# SPDX-License-Identifier: Apache-2.0
"""Conservative raw-storage estimates for capture batch admission."""

from numbers import Integral
from typing import Any

import numpy as np

# Selection planning has bounded scratch even for very long tokenized prompts.
_SELECTION_BLOCK_ROWS = 16_384
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "float32": 4,
    "float": 4,
    "float64": 8,
    "double": 8,
    "int32": 4,
    "int64": 8,
    "long": 8,
}


def _prompt_tokens(llm, prompt):
    if isinstance(prompt, dict):
        if any(
            key in prompt
            for key in ("multi_modal_data", "prompt_embeds", "encoder_prompt")
        ):
            # Multimodal preprocessing can expand token placeholders. Its final
            # geometry must be checked by worker admission, not guessed here.
            return None
        if "prompt_token_ids" in prompt:
            ids = prompt["prompt_token_ids"]
            if not isinstance(ids, (list, tuple, np.ndarray)) or any(
                isinstance(token, bool) or not isinstance(token, Integral) or token < 0
                for token in ids
            ):
                return None
            return ids
        prompt = prompt.get("prompt")
    if not isinstance(prompt, str):
        return None
    getter = getattr(llm, "get_tokenizer", None)
    if getter is None:
        return None
    try:
        return getter().encode(prompt, add_special_tokens=True)
    except (AttributeError, NotImplementedError):
        return None


def _selected_rows(tokens, select, max_tokens):
    from vllm.model_hooks.selection.host import clause_mask
    from vllm.model_hooks.selection.spec import SelectSpec

    prompt_length = len(tokens)
    # The last sampled output is returned without another model forward.
    generation_length = max(0, max_tokens - 1)
    if select is None:
        return prompt_length + generation_length
    wire = select if isinstance(select, dict) else select.to_wire()
    wire = SelectSpec.from_wire(wire).to_wire()
    count = 0
    for start in range(0, prompt_length, _SELECTION_BLOCK_ROWS):
        end = min(prompt_length, start + _SELECTION_BLOCK_ROWS)
        positions = np.arange(start, end, dtype=np.int64)
        count += int(
            clause_mask(
                wire,
                np.zeros(end - start, dtype=bool),
                positions,
                np.full(end - start, prompt_length, dtype=np.int64),
                positions - prompt_length,
                lambda start=start, end=end: np.asarray(
                    tokens[start:end], dtype=np.int64
                ),
            ).sum()
        )
    # Future token IDs are unknown: allow every possible token-ID inclusion
    # and ignore token-ID exclusions, preserving exact positional restrictions.
    generation_wire = dict(wire)
    if generation_wire.get("generation_tokens") is not None:
        generation_wire["generation"] = "all"
        generation_wire["generation_tokens"] = None
    generation_wire["exclude_generation_tokens"] = None
    for start in range(0, generation_length, _SELECTION_BLOCK_ROWS):
        end = min(generation_length, start + _SELECTION_BLOCK_ROWS)
        indices = np.arange(start, end, dtype=np.int64)
        count += int(
            clause_mask(
                generation_wire,
                np.ones(end - start, dtype=bool),
                indices + prompt_length,
                np.full(end - start, prompt_length, dtype=np.int64),
                indices,
                lambda size=end - start: np.zeros(size, dtype=np.int64),
            ).sum()
        )
    return count


def _element_size(llm, dtype, stream):
    if dtype is None:
        engine = getattr(llm, "llm_engine", None)
        config = getattr(engine, "model_config", None)
        if config is None:
            config = getattr(getattr(engine, "vllm_config", None), "model_config", None)
        dtype = getattr(config, "dtype", None)
        size = _DTYPE_BYTES.get(str(dtype).removeprefix("torch."))
        # Several MoE implementations compute router logits in float32 even
        # when the model's hidden activations use float16 or bfloat16.
        return max(4, size) if size is not None and stream == "router_logits" else size
    return _DTYPE_BYTES.get(str(dtype).removeprefix("torch."))


def _worker_row_bytes(status, layers, element_size, rank):
    layouts = status.get("layouts", {})
    shards = status.get("shards", {})
    available = set(layouts) | set(shards)
    selected = available if layers is None else set(layers)
    if not selected or not selected <= available:
        return None
    total = 0
    for layer in selected:
        shard = shards.get(layer, {})
        kind = shard.get("kind")
        # Replicated values have one owner, while attention feature shards
        # each carry a separate set of int32 request/position/token labels.
        if kind == "replicated" and rank != 0:
            continue
        if kind is None and rank != 0:
            return None
        width = layouts.get(layer, {}).get("width")
        if width is None and kind == "replicated":
            width = shard.get("global_width")
        if type(width) is not int or width < 1:
            return None
        total += width * element_size + 3 * 4
    return total


def estimate_prompt_bytes(
    llm: Any,
    prompt: Any,
    select: Any,
    stream: str,
    layers: list[int] | None,
    dtype: str | None,
    max_tokens: int,
    statuses: list[dict],
) -> int | None:
    """Estimate the raw capture budget required for one ordinary request.

    Prompt rows use vLLM's shared selector. Decode rows conservatively assume
    generation reaches ``max_tokens`` and every token predicate can match.
    Attention budgets are split equally among workers, so admission uses the
    largest worker requirement times TP size, including each shard's labels.

    Return ``None`` when final prompt geometry, storage dtype, or layer widths
    are unavailable. The caller should isolate that prompt and rely on worker
    admission. This estimates retained activation/label storage, not process
    RSS, graph pools, or caller-retained captures.
    """
    tokens = _prompt_tokens(llm, prompt)
    element_size = _element_size(llm, dtype, stream)
    if tokens is None or element_size is None or not statuses:
        return None
    per_worker = []
    for index, status in enumerate(statuses):
        rank = status.get("topology", {}).get("tp_rank", index)
        row_bytes = _worker_row_bytes(status, layers, element_size, rank)
        if row_bytes is None:
            return None
        per_worker.append(row_bytes)
    row_bytes = (
        max(per_worker) * len(statuses)
        if stream == "attention_heads"
        else sum(per_worker)
    )
    return _selected_rows(tokens, select, max_tokens) * row_bytes
