# SPDX-License-Identifier: Apache-2.0
"""Capture labelled activations from a running vLLM engine."""

from collections.abc import Iterable, Iterator
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .result import CaptureResult


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
    budget_bytes: int | None = 256 * 1024 * 1024,
    device_budget_bytes: int | None = 256 * 1024 * 1024,
    staging_bytes: int = 16 * 1024 * 1024,
    fetch_bytes: int = 16 * 1024 * 1024,
    storage_dir: str | Path | None = None,
    sample_indices: list[int] | None = None,
    **generate_kwargs,
) -> CaptureResult:
    """Capture intermediate state for a batch of prompts.

    Args:
        llm: vLLM LLM instance (compiled or eager, prefix caching on or off).
            Ordinary tensor parallelism supports eager, piecewise, and full CUDA
            graph capture. Other parallel layouts are not supported.
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
        budget_bytes: Maximum raw CPU capture storage plus pending transfer
            data, across the stream's layers. Exceeding it fails capture.
            Defaults to 256 MiB; None disables the limit. Excludes model/graph
            memory, Python objects and bounded RPC serialization temporaries.
        device_budget_bytes: Per-worker capture output/selection budget, 256 MiB
            by default. Oversized graphs use eager capture. This excludes model
            activations, KV cache and the graph pool. None disables this limit.
        staging_bytes: Reusable pinned transfer page size per worker, 16 MiB
            by default. Stored rows use pageable CPU memory.
        fetch_bytes: Target raw RPC page size, 16 MiB by default. Each page
            contains at least one row and creates bounded serialization copies.
        storage_dir: New directory receiving memory-mapped activation arrays
            and a portable manifest. None keeps the final tensors in RAM.
        sample_indices: Stable input indices, one per prompt. Defaults to local
            indices; capture_batches supplies indices in its input iterable.
        **generate_kwargs (Any): forwarded into SamplingParams.

    Returns:
        CaptureResult with exact per-sample views.
    """
    from vllm import SamplingParams
    from vllm.capture import (
        SelectSpec,
        StreamConfig,
        validate_capture_topology,
    )

    if stream not in ("hidden_states", "router_logits", "attention_heads"):
        raise ValueError(f"Unknown capture stream: {stream}")

    def to_wire(spec):
        if spec is None:
            return None
        wire = spec if isinstance(spec, dict) else spec.to_wire()
        return SelectSpec.from_wire(wire).to_wire()

    if type(fetch_bytes) is not int or fetch_bytes < 1:
        raise ValueError("fetch_bytes must be a positive integer")
    directory = Path(storage_dir) if storage_dir is not None else None
    tp_size = None

    def rpc(method, *args, **kwargs):
        results = llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)
        if tp_size is not None and len(results) != tp_size:
            raise RuntimeError(f"Capture {method} returned an incomplete worker group")
        if method in ("start_capture", "stop_capture") and not all(
            result is True for result in results
        ):
            raise RuntimeError(f"Capture {method} was not acknowledged by every worker")
        return results

    enable_kwargs: dict[str, Any] = {
        "budget_bytes": budget_bytes,
        "device_budget_bytes": device_budget_bytes,
        "staging_bytes": staging_bytes,
    }
    if layers is not None:
        enable_kwargs["layers"] = list(layers)
    if dtype is not None:
        enable_kwargs["dtype"] = dtype
    if select is not None:
        enable_kwargs["select"] = to_wire(select)
    if budget_bytes is not None:
        enable_kwargs["budget_bytes"] = budget_bytes
    # Reject invalid selections and storage configuration before any worker RPC.
    StreamConfig(**enable_kwargs)
    if generate_kwargs.get("n", 1) != 1:
        raise ValueError("Capture requires n=1 for unambiguous sample attribution")
    n_prompts = 1 if isinstance(prompts, (str, dict)) else len(prompts)
    if sample_indices is not None:
        sample_indices = CaptureResult._validate_sample_indices(
            sample_indices, n_prompts
        )

    capture_select = None
    per_prompt_selections = None
    if per_prompt_selects is not None:
        if len(per_prompt_selects) != n_prompts:
            raise ValueError(
                f"per_prompt_selects ({len(per_prompt_selects)}) must "
                f"match prompts ({n_prompts})"
            )
        per_prompt_selections = [to_wire(spec) for spec in per_prompt_selects]
        capture_select = [
            None if spec is None else {stream: spec} for spec in per_prompt_selections
        ]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=generate_kwargs.pop("temperature", 0.0),
        **generate_kwargs,
    )

    tp_size = validate_capture_topology(rpc("capture_status", stream))
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    try:
        rpc("start_capture", stream, **enable_kwargs)
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            capture_select=capture_select,
            steering=steering,
            use_tqdm=False,
        )
        statuses = rpc("capture_status", stream)
        validate_capture_topology(statuses)
        dropped = [
            (status.get("topology", {}).get("tp_rank", 0), status["tokens_dropped"])
            for status in statuses
            if status["tokens_dropped"]
        ]
        if dropped:
            raise RuntimeError(
                f"Capture discarded rows on TP workers (rank, count): {dropped}; "
                "a complete result is required. Reduce the capture batch "
                "or increase its storage budget."
            )
        from .fetch import fetch_pages

        tensors, meta, layouts = fetch_pages(
            rpc,
            stream,
            statuses,
            tp_size,
            fetch_bytes,
            directory,
        )
    except BaseException as error:
        try:
            rpc("stop_capture", stream)
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    else:
        rpc("stop_capture", stream)
    model_config = getattr(llm.llm_engine, "model_config", None)
    if model_config is None:
        model_config = getattr(
            getattr(llm.llm_engine, "vllm_config", None), "model_config", None
        )
    result = CaptureResult(
        tensors,
        meta,
        outputs,
        layouts=layouts,
        component=stream,
        model=getattr(model_config, "model", None),
        selection=to_wire(select),
        per_prompt_selections=per_prompt_selections,
        sample_indices=sample_indices,
    )
    if directory is not None:
        from .storage import write_manifest

        write_manifest(result, directory)
    return result


def capture_batches(
    llm: Any,
    prompts: Iterable,
    *,
    batch_size: int = 32,
    per_prompt_selects: Iterable[Any | None] | None = None,
    steering: Any | None = None,
    budget_bytes: int | None = 256 * 1024 * 1024,
    storage_dir: str | Path | None = None,
    **capture_kwargs,
) -> Iterator[CaptureResult]:
    """Lazily capture batches admitted by prompt count and estimated raw bytes.

    Consume each result before advancing; the producer never runs ahead of the
    consumer. Sample indices identify positions in the original input iterable.
    Unknown multimodal or component shapes are isolated into single-prompt
    batches and remain subject to the engine's enforced storage budget.
    ``storage_dir`` writes each batch into a separate memory-mapped dataset.
    """
    from .planning import estimate_prompt_bytes

    if isinstance(prompts, (str, bytes, dict)) or not isinstance(prompts, Iterable):
        raise TypeError("capture_batches requires an iterable of prompts")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if budget_bytes is not None and (type(budget_bytes) is not int or budget_bytes < 1):
        raise ValueError("budget_bytes must be a positive integer or None")
    stream = capture_kwargs.get("stream", "hidden_states")
    statuses = llm.llm_engine.collective_rpc(
        "capture_status", args=(stream,), kwargs={}
    )
    sentinel = object()
    sources = [iter(prompts)]
    if per_prompt_selects is not None:
        sources.append(iter(per_prompt_selects))
    per_prompt_steering = isinstance(steering, list)
    if per_prompt_steering:
        sources.append(iter(steering))
    batch, selections, steers, indices = [], [], [], []
    estimated = 0

    def run_batch():
        return capture(
            llm,
            batch.copy(),
            per_prompt_selects=selections.copy()
            if per_prompt_selects is not None
            else None,
            steering=steers.copy() if per_prompt_steering else steering,
            budget_bytes=budget_bytes,
            sample_indices=indices.copy(),
            storage_dir=(Path(storage_dir) / f"batch-{indices[0]:012d}")
            if storage_dir is not None
            else None,
            **capture_kwargs,
        )

    for index, items in enumerate(zip_longest(*sources, fillvalue=sentinel)):
        if any(item is sentinel for item in items):
            raise ValueError("per-prompt selections and steering must match prompts")
        prompt = items[0]
        selection = items[1] if per_prompt_selects is not None else None
        steer = items[-1] if per_prompt_steering else None
        size = (
            estimate_prompt_bytes(
                llm,
                prompt,
                selection if selection is not None else capture_kwargs.get("select"),
                stream,
                capture_kwargs.get("layers"),
                capture_kwargs.get("dtype"),
                capture_kwargs.get("max_tokens", 1),
                statuses,
            )
            if budget_bytes is not None
            else 0
        )
        if size is not None and budget_bytes is not None and size > budget_bytes:
            raise ValueError(
                f"Prompt {index} needs up to {size} raw capture bytes, exceeding "
                f"budget_bytes={budget_bytes}; select fewer rows/layers or increase the budget"
            )
        if batch and (
            len(batch) == batch_size
            or size is None
            or (budget_bytes is not None and estimated + size > budget_bytes)
        ):
            yield run_batch()
            batch.clear()
            selections.clear()
            steers.clear()
            indices.clear()
            estimated = 0
        batch.append(prompt)
        selections.append(selection)
        steers.append(steer)
        indices.append(index)
        estimated += size or 0
        if size is None:
            yield run_batch()
            batch.clear()
            selections.clear()
            steers.clear()
            indices.clear()
            estimated = 0
    if batch:
        yield run_batch()


def release_capture_cache(llm: Any) -> None:
    """Release cached capture graphs on every worker after a capture workload."""
    results = llm.llm_engine.collective_rpc("release_capture_cache", args=(), kwargs={})
    if not results or not all(result is True for result in results):
        raise RuntimeError("Capture cache release was not acknowledged by every worker")
