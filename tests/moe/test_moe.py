# SPDX-License-Identifier: Apache-2.0
"""MoE gate steering semantics and per-request routing on OLMoE-1B-7B.

One eager engine (both source scripts, test_moe_modes.py and
test_moe_slot_routing.py, booted identical engines). Canonical
moe_router modes are 'activate' / 'deactivate' (log-softmax, per-token
max+eps / min-eps); 'boost', 'soft_hard', 'suppress' and 'steermoe' are
deprecated aliases resolving onto them. Gate steering routes per
request through the same slot machinery as decoder-layer steering.

Coverage:
  - alias configs produce byte-identical outputs to canonical modes
  - a mixed layer config (activate_ids + deactivate_ids) forces the
    activated experts INTO and the deactivated experts OUT of every
    token's top-k (captured post-steering logits)
  - the no-file spec path (params expert_ids/mode, no JSON) steers all
    rows at the target layers
  - position-conditioned gate steering steers exactly those token rows
  - two requests with disjoint deactivation sets in ONE batch: every
    captured row bears exactly one signature (XOR), row counts match
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
from vllm import SamplingParams
from vllm.capture import deserialize_captured

from helpers import MOE_MODEL, steering_spec

MODEL = MOE_MODEL
with open(os.path.join(MODEL, "config.json")) as f:
    _hf_cfg = json.load(f)
NUM_LAYERS = _hf_cfg["num_hidden_layers"]
N_EXPERTS = _hf_cfg["num_experts"]
TOP_K = _hf_cfg["num_experts_per_tok"]
ALL_LAYERS = [str(layer) for layer in range(NUM_LAYERS)]

ENGINE_KWARGS = dict(
    model=MODEL,
    enable_steer_vector=True,
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.4,
    max_model_len=4096,
)

ACT = [25]  # disjoint from DEACT: on overlap, deactivation wins
DEACT = list(range(20))
X = list(range(20))
Y = list(range(20, 40))
Z = list(range(40, 60))

ALIAS_PAIRS = [
    (
        "boost",
        {"mode": "boost", "expert_ids": ACT},
        {"mode": "activate", "expert_ids": ACT},
    ),
    (
        "suppress",
        {"mode": "suppress", "expert_ids": DEACT},
        {"mode": "deactivate", "expert_ids": DEACT},
    ),
    (
        "steermoe",
        {"mode": "steermoe", "deactivate_ids": DEACT},
        {"mode": "deactivate", "expert_ids": DEACT},
    ),
]


def rpc(llm, method, *args, **kwargs):
    return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)[0]


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
        {l: {"mode": "deactivate", "expert_ids": deact_ids} for l in ALL_LAYERS},
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
    """Post-steering router logits {layer: (rows, n_experts) float32}."""
    rpc(llm, "start_capture", "router_logits")
    try:
        generate(llm, prompts_ids, specs, max_tokens=max_tokens)
        raw = rpc(llm, "fetch_captured", "router_logits")
    finally:
        rpc(llm, "stop_capture", "router_logits")
    out = deserialize_captured(raw)[0]
    return {lid: t.float().numpy() for lid, t in out.items()}


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
    others = np.setdiff1d(np.arange(N_EXPERTS), ids)
    return rows[:, ids].max(axis=-1) <= rows[:, others].min(axis=-1)


def prompt_ids(tok, text):
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok(rendered, add_special_tokens=False).input_ids


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
    @pytest.mark.parametrize(
        "alias, alias_cfg, canonical_cfg",
        ALIAS_PAIRS,
        ids=[p[0] for p in ALIAS_PAIRS],
    )
    def test_alias_output_identical(
        self, llm, ids_a, tmp_path, alias, alias_cfg, canonical_cfg
    ):
        """Deprecated mode aliases resolve byte-identically onto canon."""
        spec_alias = moe_json_spec(
            tmp_path, f"{alias}-alias", {l: alias_cfg for l in ALL_LAYERS}
        )
        spec_canon = moe_json_spec(
            tmp_path, f"{alias}-canon", {l: canonical_cfg for l in ALL_LAYERS}
        )
        out_alias = generate(llm, [ids_a], [spec_alias])[0]
        out_canon = generate(llm, [ids_a], [spec_canon])[0]
        assert out_alias == out_canon, (
            f"alias={out_alias!r} canonical={out_canon!r}"
        )

    def test_mixed_config_forces_both_directions(self, llm, ids_a, tmp_path):
        """activate_ids enter and deactivate_ids leave every top-k."""
        spec = moe_json_spec(
            tmp_path,
            "mixed",
            {
                l: {
                    "mode": "activate",
                    "activate_ids": ACT,
                    "deactivate_ids": DEACT,
                }
                for l in ALL_LAYERS
            },
        )
        logits = captured(llm, [ids_a], [spec])
        assert sorted(logits) == list(range(NUM_LAYERS)), sorted(logits)
        for lid in sorted(logits):
            order = np.argsort(logits[lid], axis=-1)[:, -TOP_K:]
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
        logits = captured(llm, [ids_a], [spec])
        assert sorted(logits) == list(range(NUM_LAYERS)), sorted(logits)
        for lid in sorted(logits):
            sig = signature(logits[lid], DEACT)
            assert sig.all(), f"L{lid}: unsteered rows {np.flatnonzero(~sig)}"

    def test_position_filter_steers_exactly_those_rows(
        self, llm, ids_a, tmp_path
    ):
        """positions on the prompt phase steer those rows and no others."""
        trig = list(range(6))
        spec = moe_json_spec(
            tmp_path,
            "trig",
            {l: {"mode": "deactivate", "expert_ids": DEACT} for l in ALL_LAYERS},
            phases=["prompt"],
            positions=trig,
        )
        logits = captured(llm, [ids_a], [spec])
        assert sorted(logits) == list(range(NUM_LAYERS)), sorted(logits)
        for lid in sorted(logits):
            rows = np.flatnonzero(signature(logits[lid], DEACT)).tolist()
            assert rows == trig, f"L{lid}: steered rows {rows} != {trig}"


class TestSlotRouting:
    """Per-request (slot-routed) gate steering in mixed batches.

    A steermoe deactivation leaves a detectable signature (see
    `signature`), so captured post-steering router logits attribute
    every token row to its config with no scheduler-order assumptions.
    """

    def test_disjoint_configs_route_per_request(
        self, llm, ids_a, ids_b, tmp_path
    ):
        """Each row bears exactly one of two disjoint signatures (XOR)."""
        spec_x = deact_spec(tmp_path, "deact-x", X)
        spec_y = deact_spec(tmp_path, "deact-y", Y)
        logits = captured(llm, [ids_a, ids_b], [spec_x, spec_y])
        la, lb = len(ids_a), len(ids_b)
        assert sorted(logits) == list(range(NUM_LAYERS)), sorted(logits)
        for lid in sorted(logits):
            sig_x = signature(logits[lid], X)
            sig_y = signature(logits[lid], Y)
            assert (
                bool(np.all(sig_x ^ sig_y))
                and int(sig_x.sum()) == la
                and int(sig_y.sum()) == lb
            ), (
                f"L{lid}: X-rows={int(sig_x.sum())}/{la} "
                f"Y-rows={int(sig_y.sum())}/{lb} "
                f"xor={bool(np.all(sig_x ^ sig_y))}"
            )

    def test_unsteered_cobatch_request_untouched(
        self, llm, ids_a, ids_b, tmp_path
    ):
        """Exactly the steered request's rows bear the signature."""
        spec_x = deact_spec(tmp_path, "deact-x", X)
        logits = captured(llm, [ids_a, ids_b], [spec_x, None])
        la, lb = len(ids_a), len(ids_b)
        for lid in sorted(logits):
            sig_x = signature(logits[lid], X)
            assert (
                int(sig_x.sum()) == la and logits[lid].shape[0] == la + lb
            ), (
                f"L{lid}: X-rows={int(sig_x.sum())}/{la} "
                f"total={logits[lid].shape[0]}/{la + lb}"
            )

    def test_slots_drain_after_completion(self, llm, ids_a, ids_b, tmp_path):
        """Config list drains and no residual steering survives release."""
        spec_z = deact_spec(tmp_path, "deact-z", Z)
        generate(llm, [ids_b], [spec_z], max_tokens=4)  # use and finish
        generate(llm, [ids_b], [None], max_tokens=4)  # post-release step
        live = rpc(llm, "list_steer_vectors")
        assert not live, f"live={live}"
        logits = captured(llm, [ids_a], [None])
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
