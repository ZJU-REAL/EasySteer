# SPDX-License-Identifier: Apache-2.0
"""Captured activations indexed by model layer and labelled sample."""

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
        first = next(iter(meta.values()), None)
        for lid, m in meta.items():
            if len(m) != layers[lid].shape[0]:
                raise RuntimeError(
                    f"layer {lid}: {len(m)} row labels for "
                    f"{layers[lid].shape[0]} rows — engine/client label desync"
                )
            if (
                m.req_ids != first.req_ids
                or not torch.equal(m.positions, first.positions)
                or not torch.equal(m.token_ids, first.token_ids)
            ):
                raise RuntimeError(
                    f"layer {lid}: capture row labels differ between layers; "
                    "sample rows cannot be aligned"
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

    def sample_rows(self, i: int, layer: int) -> torch.Tensor:
        """One sample's rows for one layer, using a view when contiguous."""
        rows = self._sample_rows[i]
        tensor = self.layers[layer]
        if not rows:
            return tensor[:0]
        start = rows[0]
        if all(row == start + offset for offset, row in enumerate(rows)):
            return tensor[start : start + len(rows)]
        return tensor[rows]

    def token(self, i: int, layer: int, position: int = -1) -> torch.Tensor:
        """One captured row by sample-relative index, not absolute token position."""
        return self.layers[layer][self._sample_rows[i][position]]

    def sample(self, i: int) -> dict[int, torch.Tensor]:
        """One sample's rows for every layer: {layer_id: (rows, dim)}."""
        return {layer: self.sample_rows(i, layer) for layer in self.layers}

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
