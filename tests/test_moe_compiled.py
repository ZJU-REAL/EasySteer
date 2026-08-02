# SPDX-License-Identifier: Apache-2.0
"""Compiled-mode (piecewise) MoE gate steering on OLMoE-1B-7B.

The vllm::steer_moe_gate op is registered as a piecewise splitting op;
under compiled execution the gate steering runs eagerly between
CUDA-graph segments. Capture streams are eager-only, so the mechanism
is verified through the steering trace instead.

Checks:
  C1  the engine boots compiled (no enforce_eager) with a moe_router
      config and generates;
  C2  steered output differs from unsteered output;
  C3  the steering trace records gate applies at every MoE layer, with
      full prompt coverage on the prefill step;
  C4  the unsteered generate leaves no apply records (routing off).

Env: GPU_ID.
"""

import glob
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))
TRACE_DIR = os.path.abspath("moe-compiled-trace")
os.environ["VLLM_STEER_TRACE_DIR"] = TRACE_DIR
os.makedirs(TRACE_DIR, exist_ok=True)
for old in glob.glob(os.path.join(TRACE_DIR, "*.jsonl")):
    os.remove(old)

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

MODEL = os.path.expanduser(
    os.environ.get("STEER_TEST_MOE_MODEL", "~/models/OLMoE-1B-7B-0125-Instruct"))

with open(os.path.join(MODEL, "config.json")) as f:
    hf_cfg = json.load(f)
NUM_LAYERS = hf_cfg["num_hidden_layers"]

llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    enforce_eager=False,
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


cfg_path = os.path.abspath("compiled_deact.json")
with open(cfg_path, "w") as f:
    json.dump({"layer_configs": {
        str(layer): {"mode": "deactivate", "expert_ids": list(range(20))}
        for layer in range(NUM_LAYERS)
    }}, f)
req = SteerVectorRequest(
    "compiled-deact", 41, steer_vector_local_path=cfg_path,
    algorithm="moe_router",
    prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])

prompt_ids = tok(
    tok.apply_chat_template(
        [{"role": "user", "content": "Count to fifteen."}],
        tokenize=False, add_generation_prompt=True),
    add_special_tokens=False).input_ids
L = len(prompt_ids)
sp = SamplingParams(temperature=0.0, max_tokens=24)


def gen(steer_req):
    outs = llm.generate({"prompt_token_ids": prompt_ids},
                        sampling_params=sp, steer_vector_request=steer_req)
    return outs[0].outputs[0].text


def read_trace():
    records = []
    for path in glob.glob(os.path.join(TRACE_DIR, "*.jsonl")):
        with open(path) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


baseline = gen(None)
n_base_applies = sum(1 for r in read_trace() if r["type"] == "apply")
steered = gen(req)
check("C1 compiled engine generated", bool(baseline) and bool(steered))
check("C2 steered differs from baseline", steered != baseline,
      f"both={steered!r}")
print(f"C2 baseline: {baseline!r}")
print(f"C2 steered : {steered!r}")

applies = [r for r in read_trace() if r["type"] == "apply"]
check("C4 no applies without steering", n_base_applies == 0,
      f"{n_base_applies} applies during unsteered run")

layers_hit = {r["layer"] for r in applies}
check("C3 applies at every MoE layer",
      layers_hit == set(range(NUM_LAYERS)),
      f"layers={sorted(layers_hit)}")
prefill_cov = {
    r["layer"]: len(r["positions"])
    for r in applies if len(r["positions"]) == L
}
check("C3 prefill covers all prompt tokens",
      set(prefill_cov) == set(range(NUM_LAYERS)),
      f"layers with full coverage: {sorted(prefill_cov)}")

for f in failures:
    print("FAIL:", f)
print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
