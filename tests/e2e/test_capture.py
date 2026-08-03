# SPDX-License-Identifier: Apache-2.0
"""Hook-based hidden-states capture on a dense model.

Covers: full-capture row counts and clear_captured; layer-subset
configs; per-sample 'last' and 'mean' reductions; per-layer row
budgets with drop reporting; dtype round-trips; router_logits on a
dense model is empty (warning, not error); fetch(clear) resets
accumulation; select clauses; row labels; per-request selection.

Capture requires enforce_eager=True and prefix caching OFF (cache-hit
tokens are never recomputed, so they cannot be captured).
"""

import os
from contextlib import contextmanager

import pytest
import torch
from vllm import SamplingParams

from vllm.capture import deserialize_captured

from helpers import DENSE_MODEL


def sel(**filters):
    """SelectSpec wire dict over both phases (the common test clause)."""
    from vllm.steer_vectors.api import SelectSpec

    return SelectSpec(phases=["prompt", "generation"], **filters).to_wire()

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
    return deserialize_captured(rpc(llm, "fetch_captured", "hidden_states"))[0]


@pytest.fixture(scope="module")
def expected_rows(llm):
    """Prefill rows + one row per decode step (max_tokens - 1 steps)."""
    prompt_tokens = len(llm.get_tokenizer().encode(TEXT))
    return prompt_tokens + SP.max_tokens - 1


def test_full_capture_all_layers_and_clear(llm, expected_rows):
    """A bare stream captures every layer fully; clear_captured drops
    rows while keeping the stream enabled."""
    with capturing(llm, "hidden_states"):
        generate(llm)
        hs = deserialize_captured(
            rpc(llm, "fetch_captured", "hidden_states", clear=False)
        )[0]
        rpc(llm, "clear_captured", "hidden_states")
        after = rpc(llm, "fetch_captured", "hidden_states")
    assert len(hs) == NUM_LAYERS, f"captured {len(hs)} layers, want {NUM_LAYERS}"
    rows = {t.shape[0] for t in hs.values()}
    assert rows == {expected_rows}, f"rows={rows}, want {{{expected_rows}}}"
    assert after == {}, "clear_captured must drop all stored rows"


def test_layer_subset_captures_only_those_layers(llm):
    with capturing(llm, "hidden_states", layers=[5, 10]):
        generate(llm)
        captured = rpc(llm, "fetch_captured", "hidden_states")
    assert sorted(captured) == [5, 10], f"got layers {sorted(captured)}"


def test_reduce_last_one_row_per_step(llm):
    with capturing(llm, "hidden_states", layers=[10], reduce="last"):
        generate(llm)
        hs = fetch(llm)
    assert hs[10].shape[0] == SP.max_tokens, (
        f"'last' stored {hs[10].shape[0]} rows, want one per step "
        f"({SP.max_tokens})"
    )


def test_reduce_mean_same_rows_different_values(llm):
    """'mean' has the same per-step row count as 'last' but other values."""
    with capturing(llm, "hidden_states", layers=[10], reduce="last"):
        generate(llm)
        last = fetch(llm)
    with capturing(llm, "hidden_states", layers=[10], reduce="mean"):
        generate(llm)
        mean = fetch(llm)
    assert mean[10].shape[0] == SP.max_tokens, f"rows={mean[10].shape[0]}"
    assert not torch.allclose(mean[10][0].float(), last[10][0].float()), (
        "'mean' prefill row equals 'last' prefill row"
    )


def test_budget_rows_caps_rows_and_reports_drops(llm):
    with capturing(llm, "hidden_states", layers=[10], budget_rows=5):
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
        with capturing(llm, "hidden_states", select=sel(tokens=[target])):
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
        with capturing(llm, "hidden_states", select=sel(positions=[0, -1])):
            generate(llm)
            states = fetch(llm)
        # Absolute position 0 and the last prompt token (-1 resolves
        # from the prompt end), regardless of decode steps.
        rows = states[10].shape[0]
        assert rows == 2, f"expected rows for positions [0, -1], got {rows}"
        with capturing(llm, "hidden_states", select=sel(positions=[0])):
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
        with capturing(llm, "hidden_states", select=sel(tokens=[target])):
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
        with pytest.raises(Exception, match="reduc"):
            rpc(llm, "start_capture", "hidden_states",
                reduce="last", select=sel(tokens=[1]))
        rpc(llm, "stop_capture", "hidden_states")


class TestWireEncoding:
    def test_bf16_raw_roundtrip(self, llm):
        """bf16 stores ship as raw bytes (no fp32 upcast) and round-trip."""
        with capturing(llm, "hidden_states", dtype="bfloat16",
                       select=sel(positions=[-1])):
            generate(llm)
            raw = rpc(llm, "fetch_captured", "hidden_states")
        layer = raw[10]
        assert layer["encoding"] == "raw"
        assert layer["dtype"] == "torch.bfloat16"
        n_vals = 1
        for d in layer["shape"]:
            n_vals *= d
        assert len(layer["data"]) == 2 * n_vals, "bf16 must ship at 2 bytes/value"
        tensor = deserialize_captured(raw)[0][10]
        assert tensor.dtype == torch.bfloat16

    def test_per_layer_fetch(self, llm):
        with capturing(llm, "hidden_states", select=sel(positions=[-1])):
            generate(llm)
            part = rpc(llm, "fetch_captured", "hidden_states",
                       layers=[3, 7], clear=True)
            assert sorted(part) == [3, 7]
            rest = rpc(llm, "fetch_captured", "hidden_states", clear=True)
        assert 3 not in rest and 7 not in rest, "fetched layers must clear"
        assert len(rest) == NUM_LAYERS - 2


class TestSelectClause:
    """The shared SelectSpec where-clause drives capture selection."""

    def test_prompt_phase_only(self, llm):
        prompt_len = len(llm.get_tokenizer().encode(TEXT))
        with capturing(llm, "hidden_states", layers=[10],
                       select={"phases": ["prompt"]}):
            generate(llm)
            hs = fetch(llm)
        assert hs[10].shape[0] == prompt_len

    def test_generation_phase_only(self, llm):
        with capturing(llm, "hidden_states", layers=[10],
                       select={"phases": ["generation"]}):
            generate(llm)
            hs = fetch(llm)
        assert hs[10].shape[0] == SP.max_tokens - 1

    def test_exclusions_subtract(self, llm, expected_rows):
        with capturing(llm, "hidden_states", layers=[10],
                       select={"phases": ["prompt", "generation"],
                               "exclude_positions": [0]}):
            generate(llm)
            hs = fetch(llm)
        assert hs[10].shape[0] == expected_rows - 1

    def test_generation_window(self, llm):
        with capturing(llm, "hidden_states", layers=[10],
                       select={"phases": ["generation"],
                               "window": [0, 2]}):
            generate(llm)
            hs = fetch(llm)
        assert hs[10].shape[0] == 2

    def test_malformed_select_rejected_at_enable(self, llm):
        with pytest.raises(Exception, match="unknown selection fields"):
            rpc(llm, "start_capture", "hidden_states",
                select={"phasez": ["prompt"]})
        rpc(llm, "stop_capture", "hidden_states")


class TestLabeledRows:
    """Every captured row carries (request id, position, token id)."""

    def test_single_request_labels(self, llm, expected_rows):
        from vllm.capture import (
            deserialize_captured,
            match_capture_request_id,
        )

        with capturing(llm, "hidden_states", layers=[10]):
            out = generate(llm)
            raw = rpc(llm, "fetch_captured", "hidden_states")
        tensors, meta = deserialize_captured(raw)
        assert meta is not None, "engine must label rows"
        labels = meta[10]
        assert len(labels) == expected_rows
        # Labels carry the engine-internal id: {client_id}-{8 hex}.
        assert len(set(labels.req_ids)) == 1
        assert match_capture_request_id(labels.req_ids[0], out.request_id)
        assert labels.positions.tolist() == list(range(expected_rows))
        assert labels.token_ids.tolist() == encoded_ids(out)

    def test_concurrent_requests_attribution(self, llm):
        """THE regression test for continuous-batching capture: rows of
        concurrently-generating requests interleave in the store, and
        the labels must attribute every row to its true request.
        (Length-based contiguous splitting silently misattributes.)"""
        from vllm.capture import (
            deserialize_captured,
            match_capture_request_id,
        )

        prompts = [
            "The capital of France is",
            "One two three four five six seven",
            "Hi",
        ]
        with capturing(llm, "hidden_states", layers=[10]):
            outs = llm.generate(prompts, sampling_params=SP, use_tqdm=False)
            raw = rpc(llm, "fetch_captured", "hidden_states")
        tensors, meta = deserialize_captured(raw)
        assert meta is not None
        labels = meta[10]

        by_req = {}
        for row, rid in enumerate(labels.req_ids):
            by_req.setdefault(rid, []).append(row)
        # One internal-id label group per request, each resolving to
        # exactly one output.
        resolved = {}
        for label in by_req:
            matches = [
                o.request_id
                for o in outs
                if match_capture_request_id(label, o.request_id)
            ]
            assert len(matches) == 1, f"label {label} matched {matches}"
            resolved[matches[0]] = label
        assert set(resolved) == {o.request_id for o in outs}

        for out in outs:
            rows = sorted(
                by_req[resolved[out.request_id]],
                key=lambda r: int(labels.positions[r]),
            )
            expected = encoded_ids(out)
            got = [int(labels.token_ids[r]) for r in rows]
            assert got == expected, (
                f"row attribution wrong for {out.request_id}: "
                f"{len(got)} rows vs {len(expected)} encoded tokens"
            )
            positions = [int(labels.positions[r]) for r in rows]
            assert positions == list(range(len(expected)))

    def test_client_split_uses_labels(self, llm):
        """The easysteer client splits by labels, exactly, under
        concurrent generation."""
        import easysteer.hidden_states as hs

        prompts = [
            "The capital of France is",
            "One two three four five six seven",
            "Hi",
        ]
        states, outs = hs.get_all_hidden_states_generate(
            llm, prompts, max_tokens=SP.max_tokens,
            layers=[10], ignore_eos=True,
        )
        assert len(states) == len(outs)
        for sample, out in zip(states, outs):
            expected = len(encoded_ids(out))
            assert sample[0].shape[0] == expected, (
                f"sample rows {sample[0].shape[0]} != encoded {expected}"
            )


class TestPerRequestSelect:
    """capture_select rides the request: each prompt can carry its own
    selection clause, resolved per request inside the batch."""

    PROMPTS = [
        "The capital of France is",
        "One two three four five six seven",
        "Hi",
    ]

    def _grouped_rows(self, labels, outs):
        from vllm.capture import match_capture_request_id

        by_req = {}
        for row, rid in enumerate(labels.req_ids):
            by_req.setdefault(rid, []).append(row)
        resolved = {}
        for label, rows in by_req.items():
            matches = [
                o.request_id
                for o in outs
                if match_capture_request_id(label, o.request_id)
            ]
            assert len(matches) == 1
            resolved[matches[0]] = sorted(
                rows, key=lambda r: int(labels.positions[r])
            )
        return resolved

    def test_per_prompt_selects_engine_side(self, llm):
        from vllm.capture import deserialize_captured
        from vllm.steer_vectors.api import SelectSpec

        first_only = SelectSpec(
            phases=["prompt", "generation"], positions=[0]
        ).to_wire()
        gen_only = SelectSpec(phases=["generation"]).to_wire()
        capture_select = [
            {"hidden_states": first_only},
            None,
            {"hidden_states": gen_only},
        ]
        with capturing(llm, "hidden_states", layers=[10]):
            outs = llm.generate(
                self.PROMPTS,
                sampling_params=SP,
                capture_select=capture_select,
                use_tqdm=False,
            )
            raw = rpc(llm, "fetch_captured", "hidden_states")
        tensors, meta = deserialize_captured(raw)
        assert meta is not None
        rows = self._grouped_rows(meta[10], outs)
        assert len(rows[outs[0].request_id]) == 1, "positions=[0] override"
        assert len(rows[outs[1].request_id]) == len(encoded_ids(outs[1])), (
            "no override falls back to the stream's global selection"
        )
        assert len(rows[outs[2].request_id]) == SP.max_tokens - 1, (
            "generation-phase override"
        )

    def test_per_request_fetch_drains(self, llm):
        from vllm.capture import deserialize_captured

        with capturing(llm, "hidden_states", layers=[10]):
            outs = llm.generate(
                self.PROMPTS, sampling_params=SP, use_tqdm=False
            )
            part_raw = rpc(
                llm, "fetch_captured", "hidden_states",
                req_ids=[outs[1].request_id], clear=True,
            )
            rest_raw = rpc(llm, "fetch_captured", "hidden_states", clear=True)
        part, part_meta = deserialize_captured(part_raw)
        rest, rest_meta = deserialize_captured(rest_raw)
        assert part[10].shape[0] == len(encoded_ids(outs[1]))
        expected_rest = sum(
            len(encoded_ids(o)) for i, o in enumerate(outs) if i != 1
        )
        assert rest[10].shape[0] == expected_rest, (
            "drained request's rows must be gone from the store"
        )

    def test_client_capture_api(self, llm):
        import easysteer.hidden_states as hs
        from vllm.steer_vectors.api import SelectSpec

        result = hs.capture(
            llm, self.PROMPTS, max_tokens=SP.max_tokens, layers=[5, 10],
            per_prompt_selects=[
                SelectSpec(phases=["prompt", "generation"], positions=[0, -1]),
                None,
                None,
            ],
            ignore_eos=True,
        )
        assert result.layer_ids == [5, 10]
        assert len(result) == 3
        assert result.sample(0)[10].shape[0] == 2, "positions [0,-1] override"
        for i in (1, 2):
            expected = len(
                list(result.outputs[i].prompt_token_ids)
                + list(result.outputs[i].outputs[0].token_ids)[:-1]
            )
            assert result.sample(i)[10].shape[0] == expected
            assert result.sample_token_ids(i) == (
                list(result.outputs[i].prompt_token_ids)
                + list(result.outputs[i].outputs[0].token_ids)[:-1]
            )

    def test_malformed_capture_select_rejected_at_admission(self, llm):
        with pytest.raises(Exception, match="unknown selection fields"):
            llm.generate(
                self.PROMPTS[0],
                sampling_params=SP,
                capture_select={"hidden_states": {"phasez": ["prompt"]}},
                use_tqdm=False,
            )
