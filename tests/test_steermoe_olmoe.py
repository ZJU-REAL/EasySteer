# SPDX-License-Identifier: Apache-2.0
"""SteerMoE (arXiv:2509.09660) runtime validation on OLMoE-1B-7B.

Exercises the full MoE path end to end on a real MoE model: router-logit
capture streams for expert detection, and gate-hook `moe_router` steering
with the paper-exact `steermoe` mode (log-softmax, activated -> max+eps,
deactivated -> min-eps, pre-top-k).

Checks:
  S1  router_logits capture: one row per prompt token on every MoE layer,
      n_experts wide;
  S2  risk-difference detection from ONE digits/words pair finds
      digit-linked experts (replicates custom_steering.ipynb);
  S3  with the steermoe deactivation config active, post-steering captured
      router logits exclude every deactivated expert from every token's
      top-k at the configured layers (deterministic mechanism check);
  S4  the same experts DO get selected without steering (S3 is not vacuous);
  S5  behavior: greedy "Count to fifteen" output changes under steering
      away from digit experts (texts reported for inspection);
  S6  the paper's precomputed faithfulness rankings for OLMoE load into a
      steermoe config and steer the counterfactual-document demo prompts
      (outputs reported; engine must survive).

Env: GPU_ID; STEERMOE_PKL optionally points at the reference
activations pickle for S6 (skipped if absent).
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))

import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.hidden_states import deserialize_hidden_states
from vllm.steer_vectors.request import SteerVectorRequest

MODEL = os.path.expanduser(
    os.environ.get("STEER_TEST_MOE_MODEL", "~/models/OLMoE-1B-7B-0125-Instruct"))
PKL = os.environ.get(
    "STEERMOE_PKL",
    "activations_[allenai--OLMoE-1B-7B-0125-Instruct]_[faithfulness].pkl",
)
N_DEACT = 40  # experts to deactivate in the digit demo (~4% of 1024)

with open(os.path.join(MODEL, "config.json")) as f:
    hf_cfg = json.load(f)
NUM_LAYERS = hf_cfg["num_hidden_layers"]
N_EXPERTS = hf_cfg["num_experts"]
TOP_K = hf_cfg["num_experts_per_tok"]
print(f"model: {NUM_LAYERS} layers, {N_EXPERTS} experts, top-{TOP_K}")

llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.4,
    max_model_len=4096,
)
tok = llm.get_tokenizer()
failures = []


def check(name, cond, detail=""):
    if not cond:
        failures.append(f"{name} {detail}")
        print("FAIL:", name, detail)
    else:
        print("OK:", name)


def rpc(method, *args, **kwargs):
    return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)[0]


def render(messages, gen_prompt):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=gen_prompt
    )


def find_sub_list(sub, seq):
    n = len(sub)
    return [
        (i, i + n - 1)
        for i in range(len(seq) - n + 1)
        if seq[i:i + n] == sub
    ]


def captured_router_logits(prompt_ids, steer_req=None):
    """Prefill `prompt_ids`, return {layer: (tokens, n_experts) float32}."""
    rpc("start_capture", "router_logits")
    llm.generate(
        {"prompt_token_ids": prompt_ids},
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        steer_vector_request=steer_req,
    )
    out = deserialize_hidden_states(rpc("fetch_captured", "router_logits"))
    rpc("stop_capture", "router_logits")
    return {lid: t.float().numpy() for lid, t in out.items()}


def topk_membership(logits_rows):
    """(tokens, n_experts) logits -> bool (tokens, n_experts) top-k mask."""
    order = np.argsort(logits_rows, axis=-1)[:, -TOP_K:]
    mask = np.zeros(logits_rows.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=-1)
    return mask


# ---------------------------------------------------------------------------
# S1/S2: detection from one contrastive pair (digits vs written numbers)
# ---------------------------------------------------------------------------
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
            {"role": "assistant", "content":
                "one, two, three, four, five, six, seven, eight, nine, ten"},
        ],
        "target":
            "one, two, three, four, five, six, seven, eight, nine, ten",
    },
}

rates = {}
for key, ex in PAIR.items():
    prompt_ids = tok(render(ex["messages"], False),
                     add_special_tokens=False).input_ids
    logits = captured_router_logits(prompt_ids)
    if key == "digits":
        check("S1 all MoE layers",
              sorted(logits) == list(range(NUM_LAYERS)),
              f"got {sorted(logits)}")
        shapes = {t.shape for t in logits.values()}
        check("S1 rows x experts",
              shapes == {(len(prompt_ids), N_EXPERTS)},
              f"shapes={shapes} expected {(len(prompt_ids), N_EXPERTS)}")

    target_ids = tok(ex["target"], add_special_tokens=False).input_ids
    spans = find_sub_list(target_ids, prompt_ids)
    assert spans, f"target span not found for {key}"
    s, e = spans[-1]
    sel = np.stack(
        [topk_membership(logits[lid][s:e + 1]) for lid in sorted(logits)]
    )  # (layer, tokens, expert)
    rates[key] = sel.mean(axis=1)  # (layer, expert) selection rate

risk_diff = rates["digits"] - rates["words"]
strong = int((risk_diff > 0.2).sum())
check("S2 digit-linked experts found", strong >= 10,
      f"only {strong} experts with risk_diff > 0.2")

flat = np.argsort(np.abs(risk_diff), axis=None)[::-1]
deact = {}  # layer -> [expert ids]
for idx in flat:
    layer, expert = divmod(int(idx), N_EXPERTS)
    if risk_diff[layer, expert] > 0:
        deact.setdefault(layer, []).append(expert)
    if sum(len(v) for v in deact.values()) == N_DEACT:
        break

cfg_path = os.path.abspath("steermoe_digits.json")
with open(cfg_path, "w") as f:
    json.dump({"layer_configs": {
        str(layer): {"mode": "deactivate", "expert_ids": ids}
        for layer, ids in deact.items()
    }}, f)
print(f"digit config: {sum(len(v) for v in deact.values())} experts "
      f"across {len(deact)} layers")

# ---------------------------------------------------------------------------
# S3/S4/S5: steer away from digit experts
# ---------------------------------------------------------------------------
digit_req = SteerVectorRequest(
    "steermoe-away-from-digits", 1,
    steer_vector_local_path=cfg_path, algorithm="moe_router",
    prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])

count_ids = tok(
    render([{"role": "user", "content": "Count to fifteen."}], True),
    add_special_tokens=False).input_ids

steered_logits = captured_router_logits(count_ids, steer_req=digit_req)
leaks = 0
for layer, ids in deact.items():
    mask = topk_membership(steered_logits[layer])
    leaks += int(mask[:, ids].sum())
check("S3 deactivated experts out of top-k", leaks == 0,
      f"{leaks} (layer, token) selections leaked through")

base_logits = captured_router_logits(count_ids)
base_hits = sum(
    int(topk_membership(base_logits[layer])[:, ids].sum())
    for layer, ids in deact.items()
)
check("S4 same experts selected without steering", base_hits > 0,
      "deactivated experts never selected at baseline; S3 is vacuous")

sp = SamplingParams(temperature=0.0, max_tokens=64)


def gen(prompt_ids, steer_req=None):
    outs = llm.generate(
        {"prompt_token_ids": prompt_ids}, sampling_params=sp,
        steer_vector_request=steer_req)
    return outs[0].outputs[0].text


baseline = gen(count_ids)
steered = gen(count_ids, digit_req)
print(f"S5 baseline : {baseline!r}")
print(f"S5 steered  : {steered!r}")
check("S5 output changes under steering", baseline != steered)
words = ("one", "two", "three", "four", "five")
print("S5 note: baseline has digits:", any(d in baseline for d in "123"),
      "| steered has number words:",
      any(w in steered.lower() for w in words))

# ---------------------------------------------------------------------------
# S6: paper's precomputed faithfulness rankings
# ---------------------------------------------------------------------------
if os.path.exists(PKL):
    import pandas as pd

    df = pd.read_pickle(PKL)
    df = df.sort_values(by="risk_diff_abs", ascending=False)
    neg = df[df["risk_diff"] < 0].head(50)  # num_experts.jsonl: 0 act/50 deact
    faith = {}
    for row in neg.itertuples():
        faith.setdefault(int(row.layer), []).append(int(row.expert))
    faith_path = os.path.abspath("steermoe_faithfulness.json")
    with open(faith_path, "w") as f:
        json.dump({"layer_configs": {
            str(layer): {"mode": "deactivate", "expert_ids": ids}
            for layer, ids in faith.items()
        }}, f)
    faith_req = SteerVectorRequest(
        "steermoe-faithfulness", 2,
        steer_vector_local_path=faith_path, algorithm="moe_router",
        prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])

    demos = [
        "Document: iPod was developed by Google\n Question: Who is the "
        "developer of iPod? \n Final Answer Only:",
        "Document: The chief executive officer of Google is Lakshmi "
        "Mittal\n Question: Who is the chief executive officer of Google? "
        "\n Final Answer Only:",
    ]
    survived = True
    for demo in demos:
        ids = tok(render([{"role": "user", "content": demo}], True),
                  add_special_tokens=False).input_ids
        try:
            print(f"S6 baseline: {gen(ids)!r}")
            print(f"S6 steered : {gen(ids, faith_req)!r}")
        except Exception as exc:  # noqa: BLE001
            survived = False
            print("S6 exception:", exc)
            break
    check("S6 precomputed rankings steer without error", survived)
else:
    print(f"S6 SKIPPED: {PKL} not found")

for f in failures:
    print("FAIL:", f)
print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
