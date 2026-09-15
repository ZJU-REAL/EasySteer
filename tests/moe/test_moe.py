# SPDX-License-Identifier: Apache-2.0
"""MoE gate steering semantics and per-request routing on OLMoE-1B-7B.

An eager engine by default exercises 'activate' / 'deactivate' (log-softmax,
per-token max+eps / min-eps). Gate steering routes per request through
the same slot machinery as decoder-layer steering. STEER_TEST_MOE_MODE selects
eager, split, or in_graph when rerunning the semantic checks in another mode.

Coverage:
  - a mixed layer config (activate_ids + deactivate_ids) forces the
    activated experts INTO and the deactivated experts OUT of every
    token's top-k (captured post-steering logits)
  - the no-file spec path (params expert_ids/mode, no JSON) steers all
    rows at the target layers
  - position-conditioned gate steering steers exactly those token rows
  - two requests with disjoint deactivation sets in ONE batch: each
    request's labeled rows bear only its own signature
  - steered + unsteered co-batch: zero contamination of the unsteered
    request
  - slots release after completion: config list drains and a subsequent
    unsteered run shows no residual signature
  - same prompt twice in one batch, one steered one not: outputs differ
"""

import json
import os

import numpy as np
import pytest
import torch
from vllm import SamplingParams

from easysteer.capture import capture

from helpers import MOE_MODEL, steering_spec

MODEL = MOE_MODEL
with open(os.path.join(MODEL, "config.json")) as f:
    _hf_cfg = json.load(f)
NUM_LAYERS = _hf_cfg["num_hidden_layers"]
N_EXPERTS = _hf_cfg["num_experts"]
TOP_K = _hf_cfg["num_experts_per_tok"]
ALL_LAYERS = [str(layer) for layer in range(NUM_LAYERS)]

ENGINE_PROFILE = "olmoe_eager"
ENGINE_KWARGS = dict(
    model=MODEL,
    worker_extension_cls="moe.worker_extension.RouterProbeWorkerExtension",
    enable_steer_vector=True,
    steer_algorithms=["moe_router"],
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.4,
    max_model_len=4096,
)
_MODE = os.environ.get("STEER_TEST_MOE_MODE")
if _MODE is not None:
    if _MODE not in ("eager", "split", "in_graph"):
        raise ValueError("STEER_TEST_MOE_MODE must be eager, split, or in_graph")
    ENGINE_PROFILE = f"olmoe_mode_{_MODE}"
    ENGINE_KWARGS.update(
        enforce_eager=_MODE == "eager",
        steer_graph_mode="split" if _MODE == "eager" else _MODE,
    )

ACT = [25]  # disjoint from DEACT: on overlap, deactivation wins
DEACT = list(range(20))
X = list(range(20))
Y = list(range(20, 40))
Z = list(range(40, 60))


def moe_json_spec(dirpath, name, layer_cfgs, **kwargs):
    """Single-vector moe_router spec backed by a layer_configs JSON."""
    path = os.path.join(str(dirpath), f"{name}.json")
    with open(path, "w") as f:
        json.dump({"layer_configs": layer_cfgs}, f)
    return steering_spec(
        source=path, algorithm="moe_router", scale=1.0, layers=None, **kwargs
    )


def deact_spec(dirpath, name, deact_ids):
    return moe_json_spec(
        dirpath,
        name,
        {
            layer: {"mode": "deactivate", "expert_ids": deact_ids}
            for layer in ALL_LAYERS
        },
    )


def generate(llm, prompts_ids, specs, max_tokens=32):
    outs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompts_ids],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        steering=specs,
        use_tqdm=False,
    )
    return [o.outputs[0].text for o in outs]


def captured(llm, prompts_ids, specs, max_tokens=1):
    """Labeled post-steering router logits from the public capture helper."""
    return capture(
        llm, [{"prompt_token_ids": ids} for ids in prompts_ids],
        stream="router_logits", max_tokens=max_tokens, temperature=0.0,
        steering=specs,
    )


def signature(rows, ids):
    """Per-row: were `ids` forced to the bottom of the expert ranking?

    Non-strict dominance: steermoe sets deactivated scores to
    per-token min - eps, but the gate logits are bf16 where eps can
    round away (ulp ~0.03-0.06 at typical log-softmax magnitudes), so
    steered rows may tie the natural minimum instead of undercutting
    it. A natural row has its ENTIRE bottom-|ids| set equal to `ids`
    with probability ~1/C(64,20), so the signature still attributes
    rows unambiguously.
    """
    rows = rows.float().numpy()
    others = np.setdiff1d(np.arange(N_EXPERTS), ids)
    return rows[:, ids].max(axis=-1) <= rows[:, others].min(axis=-1)


def prompt_ids(tok, text):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )


@pytest.fixture(scope="module")
def tok(llm):
    return llm.get_tokenizer()


@pytest.fixture(scope="module")
def ids_a(tok):
    return prompt_ids(tok, "Count to fifteen.")


@pytest.fixture(scope="module")
def ids_b(tok):
    return prompt_ids(
        tok, "Please write one short sentence about the weather in spring."
    )


class TestModeSemantics:
    def test_mixed_config_forces_both_directions(self, llm, ids_a, tmp_path):
        """activate_ids enter and deactivate_ids leave every top-k."""
        spec = moe_json_spec(
            tmp_path,
            "mixed",
            {
                layer: {
                    "mode": "activate",
                    "activate_ids": ACT,
                    "deactivate_ids": DEACT,
                }
                for layer in ALL_LAYERS
            },
        )
        logits = captured(llm, [ids_a], [spec]).layers
        assert sorted(logits) == list(range(NUM_LAYERS)), sorted(logits)
        for lid in sorted(logits):
            order = np.argsort(logits[lid].float().numpy(), axis=-1)[:, -TOP_K:]
            act_in = bool(np.isin(order, ACT).any(axis=-1).all())
            deact_out = not np.isin(order, DEACT).any()
            assert act_in and deact_out, (
                f"L{lid}: act_in={act_in} deact_out={deact_out}"
            )

    def test_no_file_spec_steers_all_rows(self, llm, ids_a):
        """params expert_ids/mode with no JSON steer every row."""
        spec = steering_spec(
            source=None,
            algorithm="moe_router",
            scale=1.0,
            layers=list(range(NUM_LAYERS)),
            params={"expert_ids": DEACT, "mode": "deactivate"},
        )
        logits = captured(llm, [ids_a], [spec]).layers
        assert sorted(logits) == list(range(NUM_LAYERS)), sorted(logits)
        for lid in sorted(logits):
            sig = signature(logits[lid], DEACT)
            assert sig.all(), f"L{lid}: unsteered rows {np.flatnonzero(~sig)}"

    def test_position_filter_steers_exactly_those_rows(self, llm, ids_a, tmp_path):
        """positions on the prompt phase steer those rows and no others."""
        trig = list(range(6))
        spec = moe_json_spec(
            tmp_path,
            "trig",
            {
                layer: {"mode": "deactivate", "expert_ids": DEACT}
                for layer in ALL_LAYERS
            },
            prompt_positions=trig,
        )
        logits = captured(llm, [ids_a], [spec]).layers
        assert sorted(logits) == list(range(NUM_LAYERS)), sorted(logits)
        for lid in sorted(logits):
            rows = np.flatnonzero(signature(logits[lid], DEACT)).tolist()
            assert rows == trig, f"L{lid}: steered rows {rows} != {trig}"

    @pytest.mark.parametrize("mode", ["soft", "soft_topk"])
    def test_soft_modes_apply_logit_spread_only_at_selected_rows(
        self, llm, ids_a, mode
    ):
        """First-layer logits expose the requested delta without recurrence."""
        token = llm.get_tokenizer().encode(" the", add_special_tokens=False)[0]
        options = dict(
            stream="router_logits", layers=[0], max_tokens=4,
            ignore_eos=True, allowed_token_ids=[token],
        )
        prompts = [{"prompt_token_ids": ids_a}]
        baseline = capture(llm, prompts, steering=False, **options)
        before = baseline.sample(0)[0]
        experts = [int(before[0].argmax())]
        experts += before[0].topk(2, largest=False).indices.tolist()
        strength = 1.5
        spec = steering_spec(
            source=None, algorithm="moe_router", scale=1.0, layers=[0],
            params={
                "mode": mode, "expert_ids": experts,
                "lambda": strength, "topk": TOP_K,
            },
            prompt_positions=[0, -1], generation="all",
        )
        result = capture(llm, prompts, steering=spec, **options)
        assert result.sample_positions(0) == baseline.sample_positions(0)
        assert result.sample_token_ids(0) == baseline.sample_token_ids(0)
        after = result.sample(0)[0]
        eligible = torch.zeros_like(before, dtype=torch.bool)
        eligible[:, experts] = True
        eligible[1:len(ids_a) - 1] = False
        if mode == "soft_topk":
            graph_mode = llm.llm_engine.vllm_config.steer_vector_config.graph_mode
            if graph_mode == "in_graph":
                order = before.argsort(dim=-1, descending=True, stable=True)
                eligible.scatter_(1, order[:, :TOP_K], False)
            else:
                # CPU/GPU topk may choose different experts at tied boundaries.
                threshold = before.topk(TOP_K, dim=-1).values[:, -1:]
                ties = before == threshold
                targeted_ties = eligible & ties
                changed_ties = targeted_ties & (after != before)
                tied_slots = TOP_K - (before > threshold).sum(dim=-1)
                targets = targeted_ties.sum(dim=-1)
                changed = changed_ties.sum(dim=-1)
                assert (changed >= (targets - tied_slots).clamp(min=0)).all()
                assert (changed <= torch.minimum(
                    targets, ties.sum(dim=-1) - tied_slots,
                )).all()
                eligible = (eligible & (before < threshold)) | changed_ties
        expected = before + eligible * before.std(dim=-1, keepdim=True) * strength
        tolerance = max(1e-5, 2 * torch.finfo(before.dtype).eps)
        torch.testing.assert_close(after, expected, rtol=tolerance, atol=tolerance)
        assert torch.equal(after[~eligible], before[~eligible])
        assert (after[eligible] > before[eligible]).all()


class TestSlotRouting:
    """Per-request (slot-routed) gate steering in mixed batches.

    A steermoe deactivation leaves a detectable signature (see
    `signature`); public capture labels assign each row to its request
    without scheduler-order assumptions.
    """

    def test_disjoint_configs_route_per_request(self, llm, ids_a, ids_b, tmp_path):
        """Each request's labeled rows bear only its own intervention."""
        spec_x = deact_spec(tmp_path, "deact-x", X)
        spec_y = deact_spec(tmp_path, "deact-y", Y)
        result = captured(llm, [ids_a, ids_b], [spec_x, spec_y])
        assert result.layer_ids == list(range(NUM_LAYERS))
        for sample, (ids, own, other) in enumerate(((ids_a, X, Y), (ids_b, Y, X))):
            assert result.sample_positions(sample) == list(range(len(ids)))
            assert result.sample_token_ids(sample) == ids
            for lid, rows in result.sample(sample).items():
                assert rows.shape == (len(ids), N_EXPERTS)
                assert signature(rows, own).all(), f"sample {sample}, L{lid}"
                assert not signature(rows, other).any(), f"sample {sample}, L{lid}"

    def test_unsteered_cobatch_request_untouched(self, llm, ids_a, ids_b, tmp_path):
        """Exactly the steered request's rows bear the signature."""
        spec_x = deact_spec(tmp_path, "deact-x", X)
        result = captured(llm, [ids_a, ids_b], [spec_x, None])
        assert result.layer_ids == list(range(NUM_LAYERS))
        for sample, ids in enumerate((ids_a, ids_b)):
            assert result.sample_positions(sample) == list(range(len(ids)))
            assert result.sample_token_ids(sample) == ids
            for lid, rows in result.sample(sample).items():
                assert rows.shape == (len(ids), N_EXPERTS)
                assert (signature(rows, X) == (sample == 0)).all(), (
                    f"sample {sample}, L{lid}: wrong request's intervention"
                )

    def test_slots_drain_after_completion(self, llm, ids_a, ids_b, tmp_path):
        """Config list drains and no residual steering survives release."""
        spec_z = deact_spec(tmp_path, "deact-z", Z)
        generate(llm, [ids_b], [spec_z], max_tokens=4)  # use and finish
        generate(llm, [ids_b], [None], max_tokens=4)  # post-release step
        live = llm.llm_engine.collective_rpc("list_steer_vectors")
        assert all(not configs for configs in live), f"live={live}"
        logits = captured(llm, [ids_a], [None]).layers
        residual = [
            (lid, ids[0])
            for lid in sorted(logits)
            for ids in (X, Y, Z)
            if signature(logits[lid], ids).any()
        ]
        assert not residual, f"residual signatures at {residual}"

    def test_same_prompt_batch_outputs_differ(self, llm, ids_a, tmp_path):
        """Per-request routing is visible at behavior level."""
        spec_x = deact_spec(tmp_path, "deact-x", X)
        outs = generate(llm, [ids_a, ids_a], [spec_x, None], max_tokens=48)
        assert outs[0] != outs[1], f"steered == unsteered: {outs[0]!r}"


def test_random_expert_choices_match_across_tensor_parallel_ranks(llm, ids_a):
    """Observe the actual gate op on every rank during prefill and decode."""
    if llm.llm_engine.vllm_config.steer_vector_config.graph_mode == "in_graph":
        pytest.skip("soft_random requires eager or split execution")

    # Exercise the engine's normal sampling path before installing the observer.
    llm.generate(
        [{"prompt_token_ids": ids_a}],
        SamplingParams(temperature=0.8, max_tokens=3, ignore_eos=True),
        steering=False, use_tqdm=False,
    )
    spec = steering_spec(
        source=None, algorithm="moe_router", layers=[0],
        params={"mode": "soft_random", "expert_ids": [0, 1, 2], "lambda": 1.5},
    )
    try:
        ranks = llm.collective_rpc("router_test_observe")
        generate(llm, [ids_a, ids_a], [spec, None], max_tokens=5)
    finally:
        records = llm.collective_rpc("router_test_collect")
    tp_size = llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size
    assert sorted(ranks) == list(range(tp_size))
    assert records[0], "the real router steering op was not observed"
    for rank, actual in zip(ranks, records):
        assert actual == records[0], f"random expert choices differ on TP rank {rank}"
    counts = [sum(row) for batch in records[0] for row in batch]
    assert 0 in counts and 3 in counts, "expected both steered and unsteered rows"
    assert set(counts) == {0, 3}, "each selected token must change three experts"
