# SPDX-License-Identifier: Apache-2.0
"""CaptureResult: labelled capture output, indexable by sample and layer.

Layers are keyed by their TRUE layer id everywhere (never positional),
and per-sample views are exact — rows are grouped by their owning
request via engine labels and ordered by sequence position, which is
the only correct grouping under continuous batching.
"""

from typing import Any, Dict, List, Optional

import torch


class CaptureResult:
    """Result of one capture call.

    Attributes:
        layers: {true_layer_id: Tensor(total_rows, dim)} in fetch order.
        outputs: the vLLM RequestOutput list, prompt order.
    """

    def __init__(
        self,
        layers: Dict[int, torch.Tensor],
        meta: Optional[Dict[int, Any]],
        outputs: Any,
    ):
        self.layers = layers
        self.outputs = outputs
        self._meta = meta
        self._sample_rows: Optional[List[List[int]]] = None
        if meta is not None:
            self._sample_rows = self._index_samples()

    @property
    def layer_ids(self) -> List[int]:
        return sorted(self.layers)

    @property
    def labelled(self) -> bool:
        return self._sample_rows is not None

    def rows(self, layer: int) -> torch.Tensor:
        return self.layers[layer]

    def meta(self, layer: int):
        """Row labels (req_ids/positions/token_ids) for a layer."""
        if self._meta is None:
            raise RuntimeError("this capture has no row labels")
        return self._meta[layer]

    def _index_samples(self) -> List[List[int]]:
        from vllm.hidden_states import match_capture_request_id

        first = self._meta[self.layer_ids[0]]
        by_label: Dict[str, List[int]] = {}
        for row, rid in enumerate(first.req_ids):
            by_label.setdefault(rid, []).append(row)
        sample_rows: List[List[int]] = []
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

    def sample(self, i: int) -> Dict[int, torch.Tensor]:
        """One sample's rows for every layer: {layer_id: (rows, dim)}."""
        if self._sample_rows is None:
            raise RuntimeError(
                "this capture has no row labels; per-sample views are "
                "unavailable"
            )
        idx = torch.tensor(self._sample_rows[i], dtype=torch.long)
        return {lid: t[idx] for lid, t in self.layers.items()}

    def sample_positions(self, i: int) -> List[int]:
        """Absolute sequence positions of sample i's rows (row order)."""
        first = self.meta(self.layer_ids[0])
        return [int(first.positions[r]) for r in self._sample_rows[i]]

    def sample_token_ids(self, i: int) -> List[int]:
        """Input token ids of sample i's rows (row order)."""
        first = self.meta(self.layer_ids[0])
        return [int(first.token_ids[r]) for r in self._sample_rows[i]]

    def to_nested(self) -> List[List[torch.Tensor]]:
        """Legacy extractor shape: [sample][layer_pos] (layers sorted by id)."""
        return [
            [self.sample(i)[lid] for lid in self.layer_ids]
            for i in range(len(self))
        ]


def capture(
    llm: Any,
    prompts: Any,
    max_tokens: int = 1,
    layers: Optional[List[int]] = None,
    dtype: Optional[str] = None,
    select: Optional[Any] = None,
    per_prompt_selects: Optional[List[Optional[Any]]] = None,
    stream: str = "hidden_states",
    **generate_kwargs,
) -> CaptureResult:
    """Capture intermediate state for a batch of prompts.

    Args:
        llm: vLLM LLM instance (enforce_eager, prefix caching off).
        prompts: prompt list (text or multimodal dicts).
        max_tokens: tokens to generate (1 = prompt-only forward).
        layers: layer-id subset (None = all hooked layers).
        dtype: engine-side storage dtype (e.g. 'float16').
        select: global SelectSpec (or wire dict) row selection.
        per_prompt_selects: one SelectSpec (or wire dict) per prompt,
            overriding the global selection for that prompt; None
            entries keep the global selection. Requires positions='all'
            semantics (no reductions).
        stream: 'hidden_states' or 'router_logits'.
        **generate_kwargs: forwarded into SamplingParams.

    Returns:
        CaptureResult with exact per-sample views.
    """
    from vllm import SamplingParams
    from vllm.hidden_states import deserialize_captured

    def to_wire(spec):
        if spec is None or isinstance(spec, dict):
            return spec
        return spec.to_wire()

    def rpc(method, *args, **kwargs):
        return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)

    enable_kwargs: Dict[str, Any] = {}
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
            None if s is None else {stream: to_wire(s)}
            for s in per_prompt_selects
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
            use_tqdm=False,
        )
        raw = rpc("fetch_captured", stream, clear=True)[0]
    finally:
        rpc("stop_capture", stream)
    tensors, meta = deserialize_captured(raw)
    return CaptureResult(tensors, meta, outputs)
