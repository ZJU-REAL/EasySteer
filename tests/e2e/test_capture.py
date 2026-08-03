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
    return llm.generate(TEXT, sampling_params=SP, use_tqdm=False)[0]


def encoded_ids(out):
    """Token ids actually fed through the model: the prompt plus every
    generated token except the last (sampled but never re-encoded)."""
    return list(out.prompt_token_ids) + list(out.outputs[0].token_ids)[:-1]


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


class TestSourceSideSelection:
    """Row selection at the hook: unneeded positions never leave the GPU."""

    def test_token_id_selection_captures_matching_rows_only(self, llm):
        """token_ids matches the whole encoded stream: prompt rows and
        decode rows (generated occurrences) alike."""
        target = llm.get_tokenizer().encode(TEXT)[2]
        with capturing(llm, "hidden_states", token_ids=[target]):
            out = generate(llm)
            states = fetch(llm)
        expected = sum(1 for t in encoded_ids(out) if t == target)
        assert expected >= 1
        assert len(states) == NUM_LAYERS
        for tensor in states.values():
            assert tensor.shape[0] == expected, (
                f"expected {expected} rows for token {target}, "
                f"got {tensor.shape[0]}"
            )

    def test_position_list_selection(self, llm, expected_rows):
        prompt_tokens = len(llm.get_tokenizer().encode(TEXT))
        with capturing(llm, "hidden_states", positions=[0, -1]):
            generate(llm)
            states = fetch(llm)
        # Absolute position 0 and the last prompt token (-1 resolves
        # from the prompt end), regardless of decode steps.
        rows = states[10].shape[0]
        assert rows == 2, f"expected rows for positions [0, -1], got {rows}"
        with capturing(llm, "hidden_states", positions=[0]):
            generate(llm)
            first_only = fetch(llm)
        with capturing(llm, "hidden_states"):
            generate(llm)
            full = fetch(llm)
        assert torch.allclose(first_only[10][0], full[10][0])
        assert full[10].shape[0] == expected_rows
        assert prompt_tokens >= 2

    def test_selection_values_match_full_capture(self, llm):
        target = llm.get_tokenizer().encode(TEXT)[-1]
        with capturing(llm, "hidden_states", token_ids=[target]):
            sel_out = generate(llm)
            selected = fetch(llm)
        with capturing(llm, "hidden_states"):
            full_out = generate(llm)
            full = fetch(llm)
        assert encoded_ids(sel_out) == encoded_ids(full_out), (
            "greedy runs diverged; row-for-row comparison is meaningless"
        )
        idx = [i for i, t in enumerate(encoded_ids(full_out)) if t == target]
        assert torch.allclose(
            selected[10], full[10][idx], atol=0
        ), "selected rows must be byte-identical to the full capture's rows"

    def test_selection_rejects_reduction_combination(self, llm):
        with pytest.raises(Exception, match="last"):
            rpc(llm, "start_capture", "hidden_states",
                positions="last", token_ids=[1])
        rpc(llm, "stop_capture", "hidden_states")


class TestWireEncoding:
    def test_bf16_raw_roundtrip(self, llm):
        """bf16 stores ship as raw bytes (no fp32 upcast) and round-trip."""
        with capturing(llm, "hidden_states", dtype="bfloat16",
                       positions=[-1]):
            generate(llm)
            raw = rpc(llm, "fetch_captured", "hidden_states")
        layer = raw[10]
        assert layer["encoding"] == "raw"
        assert layer["dtype"] == "torch.bfloat16"
        n_vals = 1
        for d in layer["shape"]:
            n_vals *= d
        assert len(layer["data"]) == 2 * n_vals, "bf16 must ship at 2 bytes/value"
        tensor = deserialize_hidden_states(raw)[10]
        assert tensor.dtype == torch.bfloat16

    def test_per_layer_fetch(self, llm):
        with capturing(llm, "hidden_states", positions=[-1]):
            generate(llm)
            part = rpc(llm, "fetch_captured", "hidden_states",
                       layers=[3, 7], clear=True)
            assert sorted(part) == [3, 7]
            rest = rpc(llm, "fetch_captured", "hidden_states", clear=True)
        assert 3 not in rest and 7 not in rest, "fetched layers must clear"
        assert len(rest) == NUM_LAYERS - 2
