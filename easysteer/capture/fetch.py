# SPDX-License-Identifier: Apache-2.0
"""Drain aligned tensor-parallel pages into their final destination."""

from pathlib import Path

import torch

from .storage import allocate_rows


def fetch_pages(rpc, stream, statuses, tp_size, fetch_bytes, directory: Path | None):
    from vllm.capture import CaptureMeta, assemble_captured

    # Old single-worker engines do not provide row counts for pagination.
    if any("layer_rows" not in status for status in statuses):
        tensors, meta, layouts = assemble_captured(
            rpc("fetch_captured", stream, clear=True), tp_size=tp_size
        )
        if directory is not None:
            for layer, rows in tensors.items():
                target = allocate_rows(rows.shape, rows.dtype, directory, layer)
                target.copy_(rows)
                tensors[layer] = target
        return tensors, meta, layouts
    layer_ids = sorted({layer for status in statuses for layer in status["layer_rows"]})
    if not any(sum(status["layer_rows"].values()) for status in statuses):
        # Fetch also validates failures recorded before the first row was stored.
        return assemble_captured(
            rpc("fetch_captured", stream, clear=True, max_rows=1), tp_size=tp_size
        )
    tensors, meta, layouts = {}, {}, {}
    for layer in layer_ids:
        counts = [status["layer_rows"].get(layer, 0) for status in statuses]
        total = max(counts)
        if any(count not in (0, total) for count in counts):
            raise RuntimeError(f"Capture layer {layer}: TP row counts disagree")
        if total == 0:
            continue
        widths = [
            status.get("layouts", {}).get(layer, {}).get("width", 0)
            for status in statuses
        ]
        # Eight bytes per value covers every supported wire dtype. TP labels
        # are duplicated. Fetch one row when discovery has no width metadata.
        page_rows = max(1, fetch_bytes // max(1, sum(widths) * 8 + tp_size * 12))
        if not all(widths):
            page_rows = 1
        offset = 0
        while offset < total:
            page, labels, page_layouts = assemble_captured(
                rpc(
                    "fetch_captured",
                    stream,
                    clear=True,
                    layers=[layer],
                    max_rows=min(page_rows, total - offset),
                ),
                tp_size=tp_size,
            )
            if set(page) != {layer} or not 0 < len(page[layer]) <= total - offset:
                raise RuntimeError(
                    f"Capture layer {layer}: inconsistent paged row counts"
                )
            rows, label = page[layer], labels[layer]
            if offset == 0:
                tensors[layer] = allocate_rows(
                    (total, rows.shape[1]),
                    rows.dtype,
                    directory,
                    layer,
                )
                meta[layer] = CaptureMeta(
                    [],
                    torch.empty(total, dtype=torch.int32),
                    torch.empty(total, dtype=torch.int32),
                )
                layouts.update(page_layouts)
                page_rows = max(
                    1,
                    fetch_bytes // (rows.shape[1] * rows.element_size() + tp_size * 12),
                )
            elif (
                rows.dtype != tensors[layer].dtype
                or rows.shape[1:] != tensors[layer].shape[1:]
                or page_layouts.get(layer) != layouts.get(layer)
            ):
                raise RuntimeError(
                    f"Capture layer {layer}: page dtype or layout changed"
                )
            end = offset + len(rows)
            tensors[layer][offset:end].copy_(rows)
            meta[layer].req_ids.extend(label.req_ids)
            meta[layer].positions[offset:end].copy_(label.positions)
            meta[layer].token_ids[offset:end].copy_(label.token_ids)
            offset = end
            del page, labels, rows, label
    return tensors, meta, layouts
