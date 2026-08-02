# SPDX-License-Identifier: Apache-2.0
"""Per-request (slot-routed) MoE steering validation on OLMoE-1B-7B.

MoE gate steering routes per request through the same slot machinery as
decoder-layer steering; the single-active limitation is gone. A steermoe
deactivation leaves a detectable signature: the deactivated experts'
scores sit STRICTLY below every other expert's on steered rows (natural
probability ~1/C(64,20) ~ 0), so captured post-steering router logits
attribute every token row to its config with no scheduler-order
assumptions.

Checks:
  R1  two requests with disjoint deactivation sets X/Y in ONE batch:
      every captured row bears exactly one signature (XOR), with row
      counts matching each request's token count, at every layer;
  R2  steered + unsteered co-batch: exactly the steered request's rows
      bear the signature — zero contamination of the unsteered request;
  R3  slots release after completion: config list drains and a
      subsequent unsteered run shows no signature anywhere;
  R4  same prompt twice in one batch, one steered one not: outputs
      differ (per-request routing visible at behavior level).

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


def make_request(name, req_id, deact_ids):
    path = os.path.abspath(f"route_{name}.json")
    with open(path, "w") as f:
        json.dump({"layer_configs": {
            str(layer): {"mode": "deactivate", "expert_ids": deact_ids}
            for layer in range(NUM_LAYERS)
        }}, f)
    return SteerVectorRequest(
        name, req_id, steer_vector_local_path=path,
        algorithm="moe_router",
        prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])


def prompt_ids(text):
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True)
    return tok(rendered, add_special_tokens=False).input_ids


def batch_generate(prompts_ids, reqs, max_tokens=1):
    outs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompts_ids],
        sampling_params=SamplingParams(temperature=0.0,
                                       max_tokens=max_tokens),
        steer_vector_request=reqs)
    return [o.outputs[0].text for o in outs]


def captured_batch(prompts_ids, reqs):
    rpc("start_capture", "router_logits")
    batch_generate(prompts_ids, reqs, max_tokens=1)
    out = deserialize_hidden_states(rpc("fetch_captured", "router_logits"))
    rpc("stop_capture", "router_logits")
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


X = list(range(0, 20))
Y = list(range(20, 40))
Z = list(range(40, 60))
req_x = make_request("deact-x", 11, X)
req_y = make_request("deact-y", 12, Y)
req_z = make_request("deact-z", 13, Z)

ids_a = prompt_ids("Count to fifteen.")
ids_b = prompt_ids("Please write one short sentence about the weather "
                   "in spring.")
LA, LB = len(ids_a), len(ids_b)

# --- R1: disjoint configs in one batch --------------------------------
logits = captured_batch([ids_a, ids_b], [req_x, req_y])
r1_ok = sorted(logits) == list(range(NUM_LAYERS))
detail = ""
for lid in sorted(logits):
    sig_x = signature(logits[lid], X)
    sig_y = signature(logits[lid], Y)
    if not (bool(np.all(sig_x ^ sig_y))
            and int(sig_x.sum()) == LA and int(sig_y.sum()) == LB):
        r1_ok = False
        detail = (f"L{lid}: X-rows={int(sig_x.sum())}/{LA} "
                  f"Y-rows={int(sig_y.sum())}/{LB} "
                  f"xor={bool(np.all(sig_x ^ sig_y))}")
        break
check("R1 disjoint configs route per-request", r1_ok, detail)

# --- R2: steered + unsteered co-batch ---------------------------------
logits = captured_batch([ids_a, ids_b], [req_x, None])
r2_ok = True
detail = ""
for lid in sorted(logits):
    sig_x = signature(logits[lid], X)
    if int(sig_x.sum()) != LA or logits[lid].shape[0] != LA + LB:
        r2_ok = False
        detail = (f"L{lid}: X-rows={int(sig_x.sum())}/{LA} "
                  f"total={logits[lid].shape[0]}/{LA + LB}")
        break
check("R2 unsteered co-batch request untouched", r2_ok, detail)

# --- R3: slots drain after completion ---------------------------------
batch_generate([ids_b], [req_z], max_tokens=4)  # use and finish a config
batch_generate([ids_b], [None], max_tokens=4)   # force a post-release step
live = rpc("list_steer_vectors")
check("R3 config list drains", not live, f"live={live}")
logits = captured_batch([ids_a], [None])
r3_clean = not any(
    signature(logits[lid], ids).any()
    for lid in sorted(logits) for ids in (X, Y, Z)
)
check("R3 no residual steering after release", r3_clean)

# --- R4: same prompt, one steered one not, one batch ------------------
outs = batch_generate([ids_a, ids_a], [req_x, None], max_tokens=48)
check("R4 same-prompt batch outputs differ", outs[0] != outs[1])
print(f"R4 steered  : {outs[0]!r}")
print(f"R4 unsteered: {outs[1]!r}")

for f in failures:
    print("FAIL:", f)
print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
