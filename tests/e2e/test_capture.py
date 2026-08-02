# SPDX-License-Identifier: Apache-2.0
"""Hook-based hidden-states capture on a dense model.

Covers: the legacy RPC shims (enable/get/clear/disable) capture all
layers with consistent row counts; layer-subset configs; per-sample
'last' and 'mean' reductions; per-layer token budgets with drop
reporting; dtype round-trips; router_logits on a dense model is empty
(warning, not error); fetch(clear) resets accumulation.

Capture requires enforce_eager=True and prefix caching OFF (cache-hit
tokens are never recomputed, so they cannot be captured).
"""

import os
from contextlib import contextmanager

import pytest
import torch
from vllm import SamplingParams

# vllm.hidden_states is a backward-compatibility alias of vllm.capture;
# imported on purpose so the alias stays covered.
from vllm.hidden_states import deserialize_hidden_states

from helpers import DENSE_MODEL

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.18,
    max_model_len=2048,
)

NUM_LAYERS = 28  # Qwen2.5-1.5B-Instruct (the default STEER_TEST_MODEL)
TEXT = "The capital of France is"
SP = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)


def rpc(llm, method, *args, **kwargs):
    return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)[0]


@contextmanager
def capturing(llm, stream, **kwargs):
    rpc(llm, "start_capture", stream, **kwargs)
    try:
        yield
    finally:
        rpc(llm, "stop_capture", stream)


def generate(llm):
    llm.generate(TEXT, sampling_params=SP, use_tqdm=False)


def fetch(llm):
    return deserialize_hidden_states(rpc(llm, "fetch_captured", "hidden_states"))


@pytest.fixture(scope="module")
def expected_rows(llm):
    """Prefill rows + one row per decode step (max_tokens - 1 steps)."""
    prompt_tokens = len(llm.get_tokenizer().encode(TEXT))
    return prompt_tokens + SP.max_tokens - 1


def test_legacy_rpc_flow_captures_all_layers(llm, expected_rows):
    """Legacy enable/get/clear/disable shims capture every layer fully."""
    rpc(llm, "enable_hidden_states_capture")
    try:
        generate(llm)
        hs = deserialize_hidden_states(rpc(llm, "get_captured_hidden_states"))
        rpc(llm, "clear_hidden_states")
    finally:
        rpc(llm, "disable_hidden_states_capture")
    assert len(hs) == NUM_LAYERS, f"captured {len(hs)} layers, want {NUM_LAYERS}"
    rows = {t.shape[0] for t in hs.values()}
    assert rows == {expected_rows}, f"rows={rows}, want {{{expected_rows}}}"


def test_layer_subset_captures_only_those_layers(llm):
    with capturing(llm, "hidden_states", layers=[5, 10]):
        generate(llm)
        captured = rpc(llm, "fetch_captured", "hidden_states")
    assert sorted(captured) == [5, 10], f"got layers {sorted(captured)}"


def test_positions_last_one_row_per_step(llm):
    with capturing(llm, "hidden_states", layers=[10], positions="last"):
        generate(llm)
        hs = fetch(llm)
    assert hs[10].shape[0] == SP.max_tokens, (
        f"'last' stored {hs[10].shape[0]} rows, want one per step "
        f"({SP.max_tokens})"
    )


def test_positions_mean_same_rows_different_values(llm):
    """'mean' has the same per-step row count as 'last' but other values."""
    with capturing(llm, "hidden_states", layers=[10], positions="last"):
        generate(llm)
        last = fetch(llm)
    with capturing(llm, "hidden_states", layers=[10], positions="mean"):
        generate(llm)
        mean = fetch(llm)
    assert mean[10].shape[0] == SP.max_tokens, f"rows={mean[10].shape[0]}"
    assert not torch.allclose(mean[10][0].float(), last[10][0].float()), (
        "'mean' prefill row equals 'last' prefill row"
    )


def test_max_tokens_budget_caps_rows_and_reports_drops(llm):
    with capturing(llm, "hidden_states", layers=[10], max_tokens=5):
        generate(llm)
        status = rpc(llm, "capture_status", "hidden_states")
        capped = fetch(llm)
    assert capped[10].shape[0] == 5, f"rows={capped[10].shape[0]}, budget=5"
    assert status["tokens_dropped"] > 0, f"no drops reported: {status}"


def test_dtype_float16_round_trips(llm):
    with capturing(llm, "hidden_states", layers=[10], dtype="float16"):
        generate(llm)
        hs = fetch(llm)
    assert hs[10].dtype == torch.float16, f"dtype={hs[10].dtype}"


def test_router_logits_empty_on_dense_model(llm):
    """A router_logits stream on a dense model yields nothing (warns)."""
    with capturing(llm, "router_logits"):
        generate(llm)
        captured = rpc(llm, "fetch_captured", "router_logits")
    assert captured == {}, f"dense model produced router logits: {captured}"


def test_fetch_clear_resets_accumulation(llm, expected_rows):
    with capturing(llm, "hidden_states", layers=[10]):
        generate(llm)
        first = fetch(llm)
        generate(llm)
        second = fetch(llm)
    assert first[10].shape[0] == expected_rows, f"rows={first[10].shape[0]}"
    assert second[10].shape[0] == expected_rows, (
        f"second fetch has {second[10].shape[0]} rows, want {expected_rows} "
        "(fetch did not clear)"
    )
