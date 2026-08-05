# SPDX-License-Identifier: Apache-2.0
"""SteerMoE (arXiv:2509.09660) runtime validation on OLMoE-1B-7B.

Exercises the full MoE path end to end on a real MoE model: router-logit
capture streams for expert detection, and gate-hook `moe_router` steering
with the paper-exact `steermoe` mode (log-softmax, activated -> max+eps,
deactivated -> min-eps, pre-top-k).

Coverage:
  - router_logits capture: one row per prompt token on every MoE layer,
    n_experts wide
  - risk-difference detection from ONE digits/words pair finds
    digit-linked experts (replicates custom_steering.ipynb)
  - with the steermoe deactivation config active, post-steering captured
    router logits exclude every deactivated expert from every token's
    top-k at the configured layers (deterministic mechanism check)
  - the same experts DO get selected without steering (the mechanism
    check is not vacuous)
  - behavior: greedy "Count to fifteen" output changes under steering
    away from digit experts
  - the paper's precomputed faithfulness rankings for OLMoE load into a
    steermoe config and steer the counterfactual-document demo prompts
    (engine must survive; skipped when the pickle is absent)
"""

import json
import os

import numpy as np
import pytest
from vllm import SamplingParams
from vllm.capture import deserialize_captured

from helpers import MOE_MODEL, steering_spec

MODEL = MOE_MODEL
# The rankings pickle ships with the official SteerMoE repo under the
# Adobe Research License (noncommercial research), so it is downloaded
# from the official repo at runtime (see replications/steermoe) and must
# never be vendored into this repo. The test skips when it is absent.
PKL = os.environ.get(
    "STEERMOE_PKL",
    "activations_[allenai--OLMoE-1B-7B-0125-Instruct]_[faithfulness].pkl",
)
N_DEACT = 100  # tuned digit->words flip count (see replications/steermoe)

with open(os.path.join(MODEL, "config.json")) as f:
    _hf_cfg = json.load(f)
NUM_LAYERS = _hf_cfg["num_hidden_layers"]
N_EXPERTS = _hf_cfg["num_experts"]
TOP_K = _hf_cfg["num_experts_per_tok"]

ENGINE_KWARGS = dict(
    model=MODEL,
    enable_steer_vector=True,
    steer_algorithms=["moe_router"],
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.4,
    max_model_len=4096,
)

# Detection material: one contrastive pair (digits vs written numbers).
PAIR = {
    "digits": {
        "messages": [
            {"role": "user", "content": "Count to ten"},
            {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"},
        ],
        "target": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
    },
    "words": {
        "messages": [
            {"role": "user", "content": "Count to ten"},
            {
                "role": "assistant",
                "content": "one, two, three, four, five, six, seven, "
                "eight, nine, ten",
            },
        ],
        "target": "one, two, three, four, five, six, seven, eight, nine, ten",
    },
}


def rpc(llm, method, *args, **kwargs):
    return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)[0]


def render(tok, messages, gen_prompt):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=gen_prompt
    )


def find_sub_list(sub, seq):
    n = len(sub)
    return [
        (i, i + n - 1)
        for i in range(len(seq) - n + 1)
        if seq[i : i + n] == sub
    ]


def captured_router_logits(llm, prompt_ids, spec=None):
    """Prefill `prompt_ids`, return {layer: (tokens, n_experts) float32}."""
    rpc(llm, "start_capture", "router_logits")
    try:
        llm.generate(
            {"prompt_token_ids": prompt_ids},
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
            steering=spec,
            use_tqdm=False,
        )
        raw = rpc(llm, "fetch_captured", "router_logits")
    finally:
        rpc(llm, "stop_capture", "router_logits")
    out = deserialize_captured(raw)[0]
    return {lid: t.float().numpy() for lid, t in out.items()}


def topk_membership(logits_rows):
    """(tokens, n_experts) logits -> bool (tokens, n_experts) top-k mask."""
    order = np.argsort(logits_rows, axis=-1)[:, -TOP_K:]
    mask = np.zeros(logits_rows.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=-1)
    return mask


def deact_json_spec(dirpath, name, layer_to_ids):
    path = os.path.join(str(dirpath), f"{name}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "layer_configs": {
                    str(layer): {"mode": "deactivate", "expert_ids": ids}
                    for layer, ids in layer_to_ids.items()
                }
            },
            f,
        )
    return steering_spec(
        source=path, algorithm="moe_router", scale=1.0, layers=None
    )


def gen(llm, prompt_ids, spec=None, max_tokens=64):
    outs = llm.generate(
        {"prompt_token_ids": prompt_ids},
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        steering=spec,
        use_tqdm=False,
    )
    return outs[0].outputs[0].text


@pytest.fixture(scope="module")
def tok(llm):
    return llm.get_tokenizer()


@pytest.fixture(scope="module")
def count_ids(tok):
    return tok(
        render(tok, [{"role": "user", "content": "Count to fifteen."}], True),
        add_special_tokens=False,
    ).input_ids


@pytest.fixture(scope="module")
def pair_captures(llm, tok):
    """{key: (prompt_ids, captured logits)} for both pair examples."""
    caps = {}
    for key, ex in PAIR.items():
        prompt_ids = tok(
            render(tok, ex["messages"], False), add_special_tokens=False
        ).input_ids
        caps[key] = (prompt_ids, captured_router_logits(llm, prompt_ids))
    return caps


@pytest.fixture(scope="module")
def risk_diff(pair_captures, tok):
    """(layer, expert) selection-rate difference over the target span."""
    rates = {}
    for key, ex in PAIR.items():
        prompt_ids, logits = pair_captures[key]
        target_ids = tok(ex["target"], add_special_tokens=False).input_ids
        spans = find_sub_list(target_ids, prompt_ids)
        assert spans, f"target span not found for {key}"
        s, e = spans[-1]
        sel = np.stack(
            [topk_membership(logits[lid][s : e + 1]) for lid in sorted(logits)]
        )  # (layer, tokens, expert)
        rates[key] = sel.mean(axis=1)  # (layer, expert) selection rate
    return rates["digits"] - rates["words"]


@pytest.fixture(scope="module")
def digit_deact(risk_diff):
    """layer -> expert ids: top-N_DEACT digit-linked experts by |risk|."""
    flat = np.argsort(np.abs(risk_diff), axis=None)[::-1]
    deact = {}
    for idx in flat:
        layer, expert = divmod(int(idx), N_EXPERTS)
        if risk_diff[layer, expert] > 0:
            deact.setdefault(layer, []).append(expert)
        if sum(len(v) for v in deact.values()) == N_DEACT:
            break
    return deact


@pytest.fixture(scope="module")
def digit_spec(digit_deact, tmp_path_factory):
    return deact_json_spec(
        tmp_path_factory.mktemp("steermoe"), "steermoe_digits", digit_deact
    )


def test_capture_all_layers_one_row_per_token(pair_captures):
    """router_logits capture covers every MoE layer, (tokens, experts)."""
    prompt_ids, logits = pair_captures["digits"]
    assert sorted(logits) == list(range(NUM_LAYERS)), f"got {sorted(logits)}"
    shapes = {t.shape for t in logits.values()}
    want = (len(prompt_ids), N_EXPERTS)
    assert shapes == {want}, f"shapes={shapes} expected {want}"


def test_detection_finds_digit_linked_experts(risk_diff):
    """One contrastive pair yields strongly digit-linked experts."""
    strong = int((risk_diff > 0.2).sum())
    assert strong >= 10, f"only {strong} experts with risk_diff > 0.2"


def test_deactivated_experts_out_of_topk(llm, count_ids, digit_deact,
                                         digit_spec):
    """Steered router logits exclude every deactivated expert from top-k."""
    steered_logits = captured_router_logits(llm, count_ids, spec=digit_spec)
    leaks = 0
    for layer, ids in digit_deact.items():
        mask = topk_membership(steered_logits[layer])
        leaks += int(mask[:, ids].sum())
    assert leaks == 0, f"{leaks} (layer, token) selections leaked through"


def test_baseline_selects_deactivated_experts(llm, count_ids, digit_deact):
    """The same experts ARE selected unsteered; the top-k check above is
    not vacuous."""
    base_logits = captured_router_logits(llm, count_ids)
    base_hits = sum(
        int(topk_membership(base_logits[layer])[:, ids].sum())
        for layer, ids in digit_deact.items()
    )
    assert base_hits > 0, (
        "deactivated experts never selected at baseline; the top-k "
        "exclusion check is vacuous"
    )


def test_output_changes_under_digit_steering(llm, count_ids, digit_spec):
    """Greedy output changes when steered away from digit experts."""
    baseline = gen(llm, count_ids)
    steered = gen(llm, count_ids, digit_spec)
    assert baseline != steered, (
        f"steering had no behavioral effect: baseline={baseline!r} "
        f"steered={steered!r}"
    )


def test_precomputed_rankings_steer_without_error(llm, tok, tmp_path):
    """The paper's released faithfulness rankings load and steer the
    counterfactual-document demo prompts; the engine must survive."""
    if not os.path.exists(PKL):
        pytest.skip(f"{PKL} not found (set STEERMOE_PKL)")
    import pandas as pd

    df = pd.read_pickle(PKL)
    df = df.sort_values(by="risk_diff_abs", ascending=False)
    neg = df[df["risk_diff"] < 0].head(50)  # num_experts.jsonl: 0 act/50 deact
    faith = {}
    for row in neg.itertuples():
        faith.setdefault(int(row.layer), []).append(int(row.expert))
    faith_spec = deact_json_spec(tmp_path, "steermoe_faithfulness", faith)

    demos = [
        (
            "Document: iPod was developed by Google\n Question: Who is the "
            "developer of iPod? \n Final Answer Only:"
        ),
        (
            "Document: The chief executive officer of Google is Lakshmi "
            "Mittal\n Question: Who is the chief executive officer of "
            "Google? \n Final Answer Only:"
        ),
    ]
    for demo in demos:
        ids = tok(
            render(tok, [{"role": "user", "content": demo}], True),
            add_special_tokens=False,
        ).input_ids
        baseline = gen(llm, ids)
        steered = gen(llm, ids, faith_spec)
        assert isinstance(baseline, str) and isinstance(steered, str)
