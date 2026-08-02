# SPDX-License-Identifier: Apache-2.0
"""MoE mode semantics on OLMoE-1B-7B (eager, single engine).

Canonical modes are 'activate' / 'deactivate' (log-softmax, per-token
max+eps / min-eps); 'boost', 'soft_hard', 'suppress' and 'steermoe' are
deprecated aliases resolving onto them.

Checks:
  A1  'boost' alias output is byte-identical to 'activate';
  A2  'suppress' alias output is byte-identical to 'deactivate';
  A3  'steermoe' + deactivate_ids is byte-identical to 'deactivate' +
      expert_ids;
  A4  a mixed layer config (activate_ids + deactivate_ids) forces the
      activated experts INTO and the deactivated experts OUT of every
      token's top-k (captured post-steering logits);
  A5  the no-file request path (moe_expert_ids/moe_mode on the request,
      no JSON) steers all rows at the target layers;
  A6  trigger-conditioned gate steering: prefill_trigger_positions
      steers exactly those token rows and no others.

Env: GPU_ID.
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))

import numpy as np
from vllm import LLM, SamplingParams
from vllm.hidden_states import deserialize_hidden_states
from vllm.steer_vectors.request import SteerVectorRequest

MODEL = os.path.expanduser(
    os.environ.get("STEER_TEST_MOE_MODEL", "~/models/OLMoE-1B-7B-0125-Instruct"))

with open(os.path.join(MODEL, "config.json")) as f:
    hf_cfg = json.load(f)
NUM_LAYERS = hf_cfg["num_hidden_layers"]
N_EXPERTS = hf_cfg["num_experts"]
TOP_K = hf_cfg["num_experts_per_tok"]

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
_next_id = [30]


def check(name, cond, detail=""):
    if not cond:
        failures.append(f"{name} {detail}")
        print("FAIL:", name, detail)
    else:
        print("OK:", name)


def rpc(method, *args, **kwargs):
    return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)[0]


def make_request(name, layer_cfgs):
    path = os.path.abspath(f"mode_{name}.json")
    with open(path, "w") as f:
        json.dump({"layer_configs": layer_cfgs}, f)
    _next_id[0] += 1
    return SteerVectorRequest(
        name, _next_id[0], steer_vector_local_path=path,
        algorithm="moe_router",
        prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])


PROMPT_IDS = tok(
    tok.apply_chat_template(
        [{"role": "user", "content": "Count to fifteen."}],
        tokenize=False, add_generation_prompt=True),
    add_special_tokens=False).input_ids
L = len(PROMPT_IDS)


def gen(steer_req=None, max_tokens=32):
    outs = llm.generate(
        {"prompt_token_ids": PROMPT_IDS},
        sampling_params=SamplingParams(temperature=0.0,
                                       max_tokens=max_tokens),
        steer_vector_request=steer_req)
    return outs[0].outputs[0].text


def captured(steer_req):
    rpc("start_capture", "router_logits")
    gen(steer_req, max_tokens=1)
    out = deserialize_hidden_states(rpc("fetch_captured", "router_logits"))
    rpc("stop_capture", "router_logits")
    return {lid: t.float().numpy() for lid, t in out.items()}


def bottom_rows(rows, ids):
    others = np.setdiff1d(np.arange(N_EXPERTS), ids)
    return rows[:, ids].max(axis=-1) <= rows[:, others].min(axis=-1)


def top_rows(rows, ids):
    others = np.setdiff1d(np.arange(N_EXPERTS), ids)
    return rows[:, ids].min(axis=-1) >= rows[:, others].max(axis=-1)


ALL = {str(layer) for layer in range(NUM_LAYERS)}
ACT = [25]  # disjoint from DEACT: on overlap, deactivation wins
DEACT = list(range(0, 20))

# --- A1-A3: alias equivalence -----------------------------------------
pairs = [
    ("A1 boost == activate",
     {l: {"mode": "boost", "expert_ids": ACT} for l in ALL},
     {l: {"mode": "activate", "expert_ids": ACT} for l in ALL}),
    ("A2 suppress == deactivate",
     {l: {"mode": "suppress", "expert_ids": DEACT} for l in ALL},
     {l: {"mode": "deactivate", "expert_ids": DEACT} for l in ALL}),
    ("A3 steermoe keys == deactivate",
     {l: {"mode": "steermoe", "deactivate_ids": DEACT} for l in ALL},
     {l: {"mode": "deactivate", "expert_ids": DEACT} for l in ALL}),
]
for name, cfg_alias, cfg_canonical in pairs:
    slug = name.split()[0]
    out_alias = gen(make_request(f"{slug}-alias", cfg_alias))
    out_canon = gen(make_request(f"{slug}-canon", cfg_canonical))
    check(name, out_alias == out_canon,
          f"alias={out_alias!r} canonical={out_canon!r}")

# --- A4: mixed activate + deactivate at one layer ---------------------
mixed_req = make_request("A4-mixed", {
    l: {"mode": "activate", "activate_ids": ACT, "deactivate_ids": DEACT}
    for l in ALL
})
logits = captured(mixed_req)
a4_ok = sorted(logits) == list(range(NUM_LAYERS))
detail = ""
for lid in sorted(logits):
    order = np.argsort(logits[lid], axis=-1)[:, -TOP_K:]
    act_in = np.isin(order, ACT).any(axis=-1).all()
    deact_out = not np.isin(order, DEACT).any()
    if not (act_in and deact_out):
        a4_ok = False
        detail = f"L{lid}: act_in={bool(act_in)} deact_out={deact_out}"
        break
check("A4 mixed config forces both directions", a4_ok, detail)

# --- A5: no-file request path -----------------------------------------
_next_id[0] += 1
nofile_req = SteerVectorRequest(
    "A5-nofile", _next_id[0], algorithm="moe_router",
    moe_expert_ids=DEACT, moe_mode="deactivate",
    target_layers=list(range(NUM_LAYERS)),
    prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
logits = captured(nofile_req)
a5_ok = all(
    bool(bottom_rows(logits[lid], DEACT).all()) for lid in sorted(logits)
)
check("A5 no-file moe_expert_ids request steers", a5_ok)

# --- A6: trigger positions --------------------------------------------
TRIG = list(range(6))
trig_cfg = {l: {"mode": "deactivate", "expert_ids": DEACT} for l in ALL}
path = os.path.abspath("mode_A6-trig.json")
with open(path, "w") as f:
    json.dump({"layer_configs": trig_cfg}, f)
_next_id[0] += 1
trig_req = SteerVectorRequest(
    "A6-trig", _next_id[0], steer_vector_local_path=path,
    algorithm="moe_router",
    prefill_trigger_positions=TRIG)
logits = captured(trig_req)
a6_ok = True
detail = ""
for lid in sorted(logits):
    rows = np.flatnonzero(bottom_rows(logits[lid], DEACT))
    if sorted(rows.tolist()) != TRIG:
        a6_ok = False
        detail = f"L{lid}: steered rows {rows.tolist()} != {TRIG}"
        break
check("A6 trigger positions steer exactly those rows", a6_ok, detail)

for f in failures:
    print("FAIL:", f)
print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
