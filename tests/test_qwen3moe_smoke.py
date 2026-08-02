# SPDX-License-Identifier: Apache-2.0
"""Qwen3-MoE architecture smoke test (single GPU, eager).

Second structural-discovery case beyond OLMoE: Qwen3-MoE passes its gate
into FusedMoE (internal-router path — the gate module is invoked inside
the MoE runner, not in the block forward), so this validates that gate
hooks fire on that call path for both steering and capture.

Checks:
  Q1  router_logits capture: every MoE layer, one row per prompt token,
      n_experts wide;
  Q2  a deactivate config forces the experts to the bottom of every
      captured row at every layer, and the engine generates normally.

Env: GPU_ID, STEER_TEST_QWEN3 (model path).
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))

import numpy as np
from vllm import LLM, SamplingParams
from vllm.hidden_states import deserialize_hidden_states
from vllm.steer_vectors.request import SteerVectorRequest

# Qwen3-30B-A3B (no suffix) is the exact model evaluated in the SteerMoE
# paper; the -2507 refresh is a later release. The base model is a hybrid
# thinking model, so prompts render with enable_thinking=False (as the
# official SteerMoE code does).
MODEL = os.environ.get(
    "STEER_TEST_QWEN3",
    "/data/zju-130/shenyl/hf/model/Qwen/Qwen3-30B-A3B",
)

with open(os.path.join(MODEL, "config.json")) as f:
    hf_cfg = json.load(f)
NUM_LAYERS = hf_cfg["num_hidden_layers"]
N_EXPERTS = hf_cfg["num_experts"]
MOE_LAYERS = [
    layer for layer in range(NUM_LAYERS)
    if layer not in (hf_cfg.get("mlp_only_layers") or [])
]
print(f"model: {NUM_LAYERS} layers ({len(MOE_LAYERS)} MoE), "
      f"{N_EXPERTS} experts")

llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=float(os.environ.get("STEER_TEST_GPU_MEM", "0.92")),
    max_model_len=2048,
    max_num_seqs=4,
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


prompt_ids = tok(
    tok.apply_chat_template(
        [{"role": "user", "content": "Count to ten."}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=False),
    add_special_tokens=False).input_ids
L = len(prompt_ids)


def captured(steer_req=None, max_tokens=1):
    rpc("start_capture", "router_logits")
    outs = llm.generate(
        {"prompt_token_ids": prompt_ids},
        sampling_params=SamplingParams(temperature=0.0,
                                       max_tokens=max_tokens),
        steer_vector_request=steer_req)
    out = deserialize_hidden_states(rpc("fetch_captured", "router_logits"))
    rpc("stop_capture", "router_logits")
    return ({lid: t.float().numpy() for lid, t in out.items()},
            outs[0].outputs[0].text)


# --- Q1: capture on the internal-router gate path ---------------------
logits, _ = captured()
check("Q1 all MoE layers captured", sorted(logits) == MOE_LAYERS,
      f"got {len(logits)} layers, expected {len(MOE_LAYERS)}")
shapes = {t.shape for t in logits.values()}
check("Q1 rows x experts", shapes == {(L, N_EXPERTS)},
      f"shapes={shapes} expected {(L, N_EXPERTS)}")

# --- Q2: deactivate steering on the internal-router gate path ---------
DEACT = list(range(10))
cfg_path = os.path.abspath("qwen3_deact.json")
with open(cfg_path, "w") as f:
    json.dump({"layer_configs": {
        str(layer): {"mode": "deactivate", "expert_ids": DEACT}
        for layer in MOE_LAYERS
    }}, f)
req = SteerVectorRequest(
    "qwen3-deact", 51, steer_vector_local_path=cfg_path,
    algorithm="moe_router",
    prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])

logits, text = captured(req, max_tokens=8)
others = np.setdiff1d(np.arange(N_EXPERTS), DEACT)
q2_ok = bool(logits) and all(
    bool((rows[:, DEACT].max(axis=-1)
          <= rows[:, others].min(axis=-1)).all())
    for rows in logits.values()
)
check("Q2 deactivate steers every row at every layer", q2_ok)
check("Q2 engine generates under steering", bool(text), repr(text))
print(f"Q2 steered output: {text!r}")

for f in failures:
    print("FAIL:", f)
print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
