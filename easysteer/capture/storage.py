# SPDX-License-Identifier: Apache-2.0
"""Portable, memory-mapped capture storage without executable object payloads."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def allocate_rows(shape, dtype, directory: Path | None, layer: int):
    shape = tuple(int(dimension) for dimension in shape)
    if directory is None:
        return torch.empty(shape, dtype=dtype)
    wire_dtype = torch.int16 if dtype == torch.bfloat16 else dtype
    array = np.lib.format.open_memmap(
        directory / f"layer-{layer}.npy",
        mode="w+",
        dtype=torch.empty((), dtype=wire_dtype).numpy().dtype,
        shape=shape,
    )
    return torch.from_numpy(array).view(dtype)


def write_manifest(result, directory: Path) -> None:
    layers = {}
    for layer in result.layer_ids:
        meta = result.meta(layer)
        np.save(directory / f"positions-{layer}.npy", meta.positions.numpy())
        np.save(directory / f"tokens-{layer}.npy", meta.token_ids.numpy())
        layers[str(layer)] = {
            "dtype": str(result.rows(layer).dtype).removeprefix("torch."),
            "req_ids": meta.req_ids,
        }
    outputs = []
    for output in result.outputs:
        record = {"request_id": output.request_id}
        for key in ("prompt", "prompt_token_ids"):
            value = getattr(output, key, None)
            if value is not None:
                record[key] = value
        record["outputs"] = [
            {
                key: getattr(item, key, None)
                for key in (
                    "index",
                    "text",
                    "token_ids",
                    "finish_reason",
                    "stop_reason",
                )
            }
            for item in getattr(output, "outputs", ())
        ]
        outputs.append(record)
    manifest = {
        "version": 1,
        "layers": layers,
        "layouts": result.layouts,
        "component": result.component,
        "model": result.model,
        "selection": result.selection,
        "per_prompt_selections": result.per_prompt_selections,
        "sample_indices": result.sample_indices,
        "outputs": outputs,
    }
    temporary = directory / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False)
    temporary.replace(directory / "manifest.json")


def save_capture(result, path) -> None:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=False)
    for layer in result.layer_ids:
        rows = result.rows(layer)
        target = allocate_rows(rows.shape, rows.dtype, directory, layer)
        target.copy_(rows)
        del target
    write_manifest(result, directory)


def load_capture(path):
    from vllm.capture import CaptureMeta
    from vllm.model_hooks.capture.serialization import resolve_storage_dtype

    from .result import CaptureResult

    directory = Path(path)
    with (directory / "manifest.json").open(encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest["version"] != 1:
        raise ValueError("Unsupported capture dataset version")
    layers, labels = {}, {}
    for key, record in manifest["layers"].items():
        layer = int(key)
        dtype = resolve_storage_dtype(record["dtype"])
        array = np.load(
            directory / f"layer-{layer}.npy", mmap_mode="c", allow_pickle=False
        )
        wire_dtype = torch.int16 if dtype == torch.bfloat16 else dtype
        expected = torch.empty((), dtype=wire_dtype).numpy().dtype
        if array.ndim != 2 or array.dtype != expected:
            raise ValueError(
                f"Capture layer {layer}: stored array dtype or shape differs"
            )
        layers[layer] = torch.from_numpy(array).view(dtype)
        positions = np.load(
            directory / f"positions-{layer}.npy", mmap_mode="c", allow_pickle=False
        )
        tokens = np.load(
            directory / f"tokens-{layer}.npy", mmap_mode="c", allow_pickle=False
        )
        if any(
            values.ndim != 1
            or len(values) != len(array)
            or values.dtype not in (np.int32, np.int64)
            for values in (positions, tokens)
        ):
            raise ValueError(f"Capture layer {layer}: invalid stored row labels")
        labels[layer] = CaptureMeta(
            req_ids=record["req_ids"],
            positions=torch.from_numpy(positions),
            token_ids=torch.from_numpy(tokens),
        )
    outputs = []
    for record in manifest["outputs"]:
        record["outputs"] = [SimpleNamespace(**item) for item in record["outputs"]]
        outputs.append(SimpleNamespace(**record))
    return CaptureResult(
        layers,
        labels,
        outputs,
        layouts={int(k): v for k, v in manifest["layouts"].items()},
        component=manifest["component"],
        model=manifest["model"],
        selection=manifest["selection"],
        per_prompt_selections=manifest.get("per_prompt_selections"),
        sample_indices=manifest["sample_indices"],
    )
