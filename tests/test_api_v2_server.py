#!/usr/bin/env python3
"""Engine-default steering via --steering-config (v2), with prefix caching.

Boots one eager engine whose default steering comes from a v2
SteeringSpec JSON (exercising the v2 branch of build_server_request in
the engine-core salt path and the worker install path), with prefix
caching enabled.

Checks:
  X0  invalid steering_config rejected at engine construction
  X1  engine boots from a SteeringSpec JSON (implies enable_steer_vector)
  X2  cold run: default slot steers every computed token (trace oracle)
  X3  warm run: prefix cache reuses blocks (num_cached_tokens oracle)
  X4  warm run: decode tokens still steered
"""

import glob
import json
import os
import shutil
import sys
import tempfile

GPU_ID = os.environ.get("GPU_ID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
TRACE_DIR = tempfile.mkdtemp(prefix="steer_trace_v2srv_")
os.environ["VLLM_STEER_TRACE_DIR"] = TRACE_DIR

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec  # noqa: E402

MODEL = os.environ.get(
    "STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"
)
VECTOR = os.environ.get(
    "STEER_TEST_VECTOR",
    os.path.expanduser("~/EasySteer/vectors/happy_diffmean.gguf"),
)
LAYER = 10
PROMPT = list(range(100, 100 + 48))

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"OK: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        FAILURES.append(name)


# --- X0: invalid config rejected before boot ---------------------------
try:
    LLM(model=MODEL, steering_config='{"vectors": []}')
except Exception as e:
    check("X0 invalid steering_config rejected at construction",
          "non-empty" in str(e), f"message: {e}")
else:
    check("X0 invalid steering_config rejected at construction", False,
          "did not raise")

SPEC_JSON = SteeringSpec(
    vectors=[
        VectorSpec(
            source=VECTOR,
            scale=0.5,
            layers=[LAYER],
            apply=ApplySpec(phases=["prompt", "generation"]),
        )
    ]
).model_dump_json()

llm = LLM(
    model=MODEL,
    steering_config=SPEC_JSON,
    enforce_eager=True,
    enable_prefix_caching=True,
    max_model_len=512,
    max_num_seqs=4,
    gpu_memory_utilization=0.18,
)
check("X1 engine boots from SteeringSpec JSON", True)
params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)


def read_trace(min_step):
    steps, applies = {}, []
    for path in glob.glob(os.path.join(TRACE_DIR, "*.jsonl")):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["step"] <= min_step:
                    continue
                if rec["type"] == "step":
                    steps[rec["step"]] = rec
                elif rec["type"] == "apply" and rec["layer"] == LAYER:
                    applies.append(rec)
    return steps, applies


def run(prompt_ids):
    start = max(read_trace(0)[0], default=0)
    outs = llm.generate(
        TokensPrompt(prompt_token_ids=list(prompt_ids)),
        params,
        use_tqdm=False,
    )
    steps, applies = read_trace(start)
    abs_pos = set()
    for rec in applies:
        step = steps[rec["step"]]
        num_computed = step["num_computed"][0]
        abs_pos.update(pos + num_computed for pos in rec["positions"])
    return outs[0], abs_pos


out_cold, pos_cold = run(PROMPT)
check(
    "X2 cold run: default slot steers all computed tokens",
    pos_cold == set(range(48 + 3)),
    f"missing {sorted(set(range(51)) - pos_cold)[:5]}, "
    f"extra {sorted(pos_cold - set(range(51)))[:5]}",
)
check(
    "X2b cold run had no cache hits",
    out_cold.num_cached_tokens == 0,
    f"got {out_cold.num_cached_tokens}",
)

out_warm, pos_warm = run(PROMPT)
check(
    "X3 warm run reuses cached blocks under the v2 server salt",
    0 < out_warm.num_cached_tokens <= 48,
    f"got {out_warm.num_cached_tokens}",
)
check(
    "X4 warm run still steers the computed suffix and decode tokens",
    {48, 49, 50} <= pos_warm,
    f"got {sorted(pos_warm)}",
)

shutil.rmtree(TRACE_DIR, ignore_errors=True)
print(f"\nOVERALL: {'FAIL' if FAILURES else 'PASS'}")
sys.exit(1 if FAILURES else 0)
