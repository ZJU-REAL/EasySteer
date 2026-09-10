# SPDX-License-Identifier: Apache-2.0
"""Capture output indexed by model layer id and sample.

Per-sample views group rows by engine request labels and order them
by sequence position.
"""

from typing import Any

import torch


class CaptureResult:
    """Result of one capture call.

    Attributes:
        layers: {true_layer_id: Tensor(total_rows, dim)} in fetch order.
        outputs: the vLLM RequestOutput list, prompt order.
        layouts: component dimensions by layer. Attention-head outputs include
            width, query num_heads, and per-head value-output head_size.
    """

    def __init__(
        self,
        layers: dict[int, torch.Tensor],
        meta: dict[int, Any],
        outputs: Any,
        layouts: dict[int, dict[str, int]] | None = None,
    ):
        self.layers = layers
        self.outputs = outputs
        self.layouts = layouts or {}
        for lid, layout in self.layouts.items():
            if lid not in layers or layers[lid].shape[-1] != layout["width"]:
                raise ValueError(f"layer {lid}: capture layout does not match rows")
        if not isinstance(meta, dict) or set(meta) != set(layers):
            raise ValueError("capture requires row labels for every captured layer")
        self._meta = meta
        for lid, m in meta.items():
            if len(m) != layers[lid].shape[0]:
                raise RuntimeError(
                    f"layer {lid}: {len(m)} row labels for "
                    f"{layers[lid].shape[0]} rows — engine/client label desync"
                )
        self._sample_rows = self._index_samples()

    @property
    def layer_ids(self) -> list[int]:
        return sorted(self.layers)

    @property
    def labelled(self) -> bool:
        return True

    def rows(self, layer: int) -> torch.Tensor:
        return self.layers[layer]

    def meta(self, layer: int):
        """Row labels (req_ids/positions/token_ids) for a layer."""
        return self._meta[layer]

    def _index_samples(self) -> list[list[int]]:
        from vllm.capture import match_capture_request_id

        if not self.layers:
            return [[] for _ in self.outputs]
        first = self._meta[self.layer_ids[0]]
        by_label: dict[str, list[int]] = {}
        for row, rid in enumerate(first.req_ids):
            by_label.setdefault(rid, []).append(row)
        sample_rows: list[list[int]] = []
        claimed = set()
        for output in self.outputs:
            matches = [
                label
                for label in by_label
                if label not in claimed
                and match_capture_request_id(label, output.request_id)
            ]
            if len(matches) > 1:
                raise RuntimeError(
                    f"request {output.request_id!r} matches several row "
                    f"label groups {matches!r}; duplicate client request "
                    "ids cannot be attributed"
                )
            if matches:
                claimed.add(matches[0])
                rows = by_label[matches[0]]
                rows.sort(key=lambda r: int(first.positions[r]))
                sample_rows.append(rows)
            else:
                sample_rows.append([])
        stale = set(by_label) - claimed
        if stale:
            raise RuntimeError(
                f"captured rows belong to requests outside this call: "
                f"{sorted(stale)[:5]} — the capture store was stale"
            )
        return sample_rows

    def __len__(self) -> int:
        return len(self.outputs)

    def sample(self, i: int) -> dict[int, torch.Tensor]:
        """One sample's rows for every layer: {layer_id: (rows, dim)}."""
        idx = torch.tensor(self._sample_rows[i], dtype=torch.long)
        return {lid: t[idx] for lid, t in self.layers.items()}

    def sample_positions(self, i: int) -> list[int]:
        """Absolute sequence positions of sample i's rows (row order)."""
        if not self.layers:
            return []
        first = self.meta(self.layer_ids[0])
        return [int(first.positions[r]) for r in self._sample_rows[i]]

    def sample_token_ids(self, i: int) -> list[int]:
        """Input token ids of sample i's rows (row order)."""
        if not self.layers:
            return []
        first = self.meta(self.layer_ids[0])
        return [int(first.token_ids[r]) for r in self._sample_rows[i]]

    def to_nested(self) -> list[list[torch.Tensor]]:
        """Return `[sample][layer_pos]` tensors with layers sorted by model id."""
        layer_ids = self.layer_ids
        samples = []
        for i in range(len(self)):
            sample = self.sample(i)
            samples.append([sample[lid] for lid in layer_ids])
        return samples


def capture(
    llm: Any,
    prompts: Any,
    max_tokens: int = 1,
    layers: list[int] | None = None,
    dtype: str | None = None,
    select: Any | None = None,
    per_prompt_selects: list[Any | None] | None = None,
    stream: str = "hidden_states",
    steering: Any | None = None,
    **generate_kwargs,
) -> CaptureResult:
    """Capture intermediate state for a batch of prompts.

    Args:
        llm: Single-worker vLLM LLM instance (compiled or eager,
            prefix caching on or off). Tensor-parallel capture is not supported.
        prompts: prompt list (text or multimodal dicts).
        max_tokens: tokens to generate (1 = prompt-only forward).
        layers: layer-id subset (None = all hooked layers).
        dtype: engine-side storage dtype (e.g. 'float16').
        select: global SelectSpec (or wire dict) row selection.
        per_prompt_selects: one SelectSpec (or wire dict) per prompt,
            overriding the global selection for that prompt; None
            entries keep the global selection. The helper returns selected rows
            without engine-side reduction.
        stream: 'hidden_states', 'router_logits', or 'attention_heads'.
        steering: SteeringSpec (or per-prompt list), as in LLM.generate().
        **generate_kwargs (Any): forwarded into SamplingParams.

    Returns:
        CaptureResult with exact per-sample views.
    """
    from vllm import SamplingParams
    from vllm.capture import deserialize_captured

    def to_wire(spec):
        if spec is None or isinstance(spec, dict):
            return spec
        return spec.to_wire()

    def rpc(method, *args, **kwargs):
        results = llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)
        if len(results) != 1:
            raise RuntimeError(
                f"capture expects a single worker, got {len(results)} "
                "RPC results — tensor-parallel capture would return "
                "per-rank shards and is not supported"
            )
        return results

    enable_kwargs: dict[str, Any] = {}
    if layers is not None:
        enable_kwargs["layers"] = list(layers)
    if dtype is not None:
        enable_kwargs["dtype"] = dtype
    if select is not None:
        enable_kwargs["select"] = to_wire(select)

    capture_select = None
    if per_prompt_selects is not None:
        if len(per_prompt_selects) != len(prompts):
            raise ValueError(
                f"per_prompt_selects ({len(per_prompt_selects)}) must "
                f"match prompts ({len(prompts)})"
            )
        capture_select = [
            None if s is None else {stream: to_wire(s)} for s in per_prompt_selects
        ]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=generate_kwargs.pop("temperature", 0.0),
        **generate_kwargs,
    )

    rpc("start_capture", stream, **enable_kwargs)
    try:
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            capture_select=capture_select,
            steering=steering,
            use_tqdm=False,
        )
        raw = rpc("fetch_captured", stream, clear=True)[0]
    finally:
        rpc("stop_capture", stream)
    tensors, meta = deserialize_captured(raw)
    layouts = {lid: info["layout"] for lid, info in raw.items() if "layout" in info}
    return CaptureResult(tensors, meta, outputs, layouts=layouts)
