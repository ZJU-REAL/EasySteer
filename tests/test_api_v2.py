#!/usr/bin/env python3
"""GPU trace-oracle validation of the v2 steering API (STEERING_API_V2.md).

Boots one eager engine with chunked prefill (64-token chunks) and uses
the steering trace as an exact oracle for which absolute positions were
steered by `llm.generate(..., steering=SteeringSpec(...))`.

Checks:
  W1  phases=["prompt"] covers every prompt token across chunks
  W2  phases=["generation"] steers only decode tokens
  W3  exclusions compose with phase-wide prompt selection (v1 bypass gone)
  W4  positions=[-1] fires once at the last prompt token
  W5  generation_window=(0,2) steers exactly decode steps 0 and 1
  W6  generation_window=(1,None) skips only decode step 0
  W7  token filter selects exactly the matching prompt positions
  W8  multi-vector: per-vector apply clauses act independently per layer
  W9  deprecated steer_vector_request path still works
  W10 steering= and steer_vector_request= together are rejected
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
TRACE_DIR = tempfile.mkdtemp(prefix="steer_trace_v2_")
os.environ["VLLM_STEER_TRACE_DIR"] = TRACE_DIR

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.steer_vectors import (  # noqa: E402
    ApplySpec,
    SteeringSpec,
    VectorSpec,
)
from vllm.steer_vectors.request import SteerVectorRequest  # noqa: E402

MODEL = os.environ.get(
    "STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"
)
VECTOR = os.environ.get(
    "STEER_TEST_VECTOR",
    os.path.expanduser("~/EasySteer/vectors/happy_diffmean.gguf"),
)
LAYER = 10
LAYER_B = 12
PROMPT_LONG = list(range(100, 100 + 129))  # chunks of 64, 64, 1

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"OK: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        FAILURES.append(name)


def spec(layers=None, **apply_kwargs):
    return SteeringSpec(
        vectors=[
            VectorSpec(
                source=VECTOR,
                scale=0.5,
                layers=layers or [LAYER],
                apply=ApplySpec(**apply_kwargs),
            )
        ]
    )


def read_trace(min_step, layer):
    steps, applies = {}, []
    for path in glob.glob(os.path.join(TRACE_DIR, "*.jsonl")):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["step"] <= min_step:
                    continue
                if rec["type"] == "step":
                    steps[rec["step"]] = rec
                elif rec["type"] == "apply" and rec["layer"] == layer:
                    applies.append(rec)
    return steps, applies


def max_step():
    steps, _ = read_trace(0, LAYER)
    return max(steps, default=0)


def steered_abs(steps, applies):
    out = []
    for rec in applies:
        step = steps[rec["step"]]
        assert len(step["req_ids"]) == 1, "test expects single-request batches"
        num_computed = step["num_computed"][0]
        is_prefill_step = step["num_output"][0] == 0
        for pos in rec["positions"]:
            out.append((pos + num_computed, is_prefill_step))
    return out


llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    enforce_eager=True,
    enable_prefix_caching=False,
    enable_chunked_prefill=True,
    max_num_batched_tokens=64,
    max_num_seqs=4,
    max_model_len=512,
    gpu_memory_utilization=0.18,
)
params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)


def run(steering=None, layer=LAYER, **gen_kwargs):
    start = max_step()
    llm.generate(
        TokensPrompt(prompt_token_ids=list(PROMPT_LONG)),
        params,
        steering=steering,
        use_tqdm=False,
        **gen_kwargs,
    )
    steps, applies = read_trace(start, layer)
    return steps, steered_abs(steps, applies)


# --- W1: prompt phase, chunked coverage --------------------------------
steps, pos = run(spec(phases=["prompt"]))
abs_set = {p for p, _ in pos}
check(
    "W1 phases=['prompt'] covers all 129 prompt tokens across chunks",
    abs_set == set(range(129)) and all(is_pf for _, is_pf in pos),
    f"missing {sorted(set(range(129)) - abs_set)[:5]}",
)

# --- W2: generation phase only -----------------------------------------
steps, pos = run(spec(phases=["generation"]))
abs_set = {p for p, _ in pos}
check(
    "W2 phases=['generation'] steers only decode tokens",
    abs_set == {129, 130, 131} and all(not is_pf for _, is_pf in pos),
    f"got {sorted(abs_set)}",
)

# --- W3: exclusions compose with phase-wide selection ------------------
steps, pos = run(spec(phases=["prompt"], exclude_positions=[0, -1]))
abs_set = {p for p, _ in pos}
check(
    "W3 exclude_positions=[0,-1] removes first and last prompt token",
    abs_set == set(range(1, 128)),
    f"got extra {sorted(abs_set - set(range(1, 128)))[:5]}, "
    f"missing {sorted(set(range(1, 128)) - abs_set)[:5]}",
)

# --- W4: negative position ---------------------------------------------
steps, pos = run(spec(phases=["prompt"], positions=[-1]))
check(
    "W4 positions=[-1] fires once at abs 128",
    sorted(p for p, _ in pos) == [128],
    f"got {sorted(p for p, _ in pos)}",
)

# --- W5/W6: exact generation windows -----------------------------------
steps, pos = run(spec(phases=["generation"], generation_window=(0, 2)))
check(
    "W5 window (0,2) steers exactly decode steps 0 and 1 (abs 129,130)",
    sorted(p for p, _ in pos) == [129, 130],
    f"got {sorted(p for p, _ in pos)}",
)
steps, pos = run(spec(phases=["generation"], generation_window=(1, None)))
check(
    "W6 window (1,None) skips only decode step 0 (abs 130,131)",
    sorted(p for p, _ in pos) == [130, 131],
    f"got {sorted(p for p, _ in pos)}",
)

# --- W7: token filter ---------------------------------------------------
steps, pos = run(spec(phases=["prompt"], tokens=[100, 150]))
check(
    "W7 token filter hits exactly the matching prompt positions",
    sorted(p for p, _ in pos) == [0, 50],
    f"got {sorted(p for p, _ in pos)}",
)

# --- W8: multi-vector with independent apply clauses -------------------
multi = SteeringSpec(
    vectors=[
        VectorSpec(
            source=VECTOR,
            scale=0.5,
            layers=[LAYER],
            apply=ApplySpec(phases=["prompt"], positions=[0]),
        ),
        VectorSpec(
            source=VECTOR,
            scale=0.5,
            layers=[LAYER_B],
            apply=ApplySpec(phases=["generation"]),
        ),
    ]
)
start = max_step()
llm.generate(
    TokensPrompt(prompt_token_ids=list(PROMPT_LONG)),
    params,
    steering=multi,
    use_tqdm=False,
)
steps_a, applies_a = read_trace(start, LAYER)
steps_b, applies_b = read_trace(start, LAYER_B)
pos_a = sorted(p for p, _ in steered_abs(steps_a, applies_a))
pos_b = sorted(p for p, _ in steered_abs(steps_b, applies_b))
check(
    "W8 multi-vector: layer 10 prompt pos 0 / layer 12 decode tokens",
    pos_a == [0] and pos_b == [129, 130, 131],
    f"layer10={pos_a} layer12={pos_b}",
)

# --- W9: deprecated v1 path still works --------------------------------
v1_req = SteerVectorRequest(
    steer_vector_name="v1-compat",
    steer_vector_int_id=901,
    steer_vector_local_path=VECTOR,
    scale=0.5,
    target_layers=[LAYER],
    prefill_trigger_positions=[-1],
)
steps, pos = run(None, steer_vector_request=v1_req)
check(
    "W9 deprecated steer_vector_request still steers (abs 128)",
    sorted(p for p, _ in pos) == [128],
    f"got {sorted(p for p, _ in pos)}",
)

# --- W10: mixing v1 and v2 arguments rejected --------------------------
try:
    llm.generate(
        TokensPrompt(prompt_token_ids=list(PROMPT_LONG)),
        params,
        steering=spec(phases=["prompt"]),
        steer_vector_request=v1_req,
        use_tqdm=False,
    )
except ValueError as e:
    check("W10 steering + steer_vector_request rejected", "not both" in str(e))
else:
    check("W10 steering + steer_vector_request rejected", False, "did not raise")

shutil.rmtree(TRACE_DIR, ignore_errors=True)
print(f"\nOVERALL: {'FAIL' if FAILURES else 'PASS'}")
sys.exit(1 if FAILURES else 0)
