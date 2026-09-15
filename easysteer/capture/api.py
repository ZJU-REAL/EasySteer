# SPDX-License-Identifier: Apache-2.0
"""Capture labelled activations from a running vLLM engine."""

from collections.abc import Iterator, Sequence
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
    budget_bytes: int | None = None,
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
            Excludes model/graph memory and RPC serialization temporaries.
        **generate_kwargs (Any): forwarded into SamplingParams.

    Returns:
        CaptureResult with exact per-sample views.
    """
    from vllm import SamplingParams
    from vllm.capture import (
        SelectSpec,
        StreamConfig,
        assemble_captured,
        validate_capture_topology,
    )

    if stream not in ("hidden_states", "router_logits", "attention_heads"):
        raise ValueError(f"Unknown capture stream: {stream}")

    def to_wire(spec):
        if spec is None:
            return None
        wire = spec if isinstance(spec, dict) else spec.to_wire()
        return SelectSpec.from_wire(wire).to_wire()

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

    enable_kwargs: dict[str, Any] = {}
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

    tp_size = validate_capture_topology(rpc("capture_status", stream))
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
        tensors, meta, layouts = assemble_captured(
            rpc("fetch_captured", stream, clear=True), tp_size=tp_size
        )
    except BaseException as error:
        try:
            rpc("stop_capture", stream)
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    else:
        rpc("stop_capture", stream)
    return CaptureResult(tensors, meta, outputs, layouts=layouts)


def capture_batches(
    llm: Any,
    prompts: Sequence,
    *,
    batch_size: int = 32,
    per_prompt_selects: list[Any | None] | None = None,
    steering: Any | None = None,
    budget_bytes: int | None = 256 * 1024 * 1024,
    **capture_kwargs,
) -> Iterator[CaptureResult]:
    """Yield captured batches in prompt order, releasing worker storage each time.

    Consume or save each result before advancing instead of retaining the full
    iterator as a list. Result sample indices are local to each yielded batch.
    Per-prompt selections and steering lists follow the same batch boundaries.
    The byte budget has the same scope as :func:`capture`; a single large
    batch can still exceed it. Compatible capture graphs are reused.
    """
    if isinstance(prompts, (str, bytes, dict)) or not isinstance(prompts, Sequence):
        raise TypeError("capture_batches requires a sequence of prompts")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if per_prompt_selects is not None and len(per_prompt_selects) != len(prompts):
        raise ValueError("per_prompt_selects length must match prompts")
    if isinstance(steering, list) and len(steering) != len(prompts):
        raise ValueError("steering length must match prompts")
    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        yield capture(
            llm,
            prompts[start:end],
            per_prompt_selects=(
                per_prompt_selects[start:end]
                if per_prompt_selects is not None
                else None
            ),
            steering=steering[start:end] if isinstance(steering, list) else steering,
            budget_bytes=budget_bytes,
            **capture_kwargs,
        )
