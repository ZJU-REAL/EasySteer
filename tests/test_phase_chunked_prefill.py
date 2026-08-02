#!/usr/bin/env python3
"""Phase classification and chunked-prefill correctness.

Uses the steering trace as an exact oracle for which absolute token
positions were steered. Covers the cases the old query-length heuristic
misclassified (1-token prompts, 1-token chunked-prefill tails), the
prompt-relative semantics of negative trigger positions under chunking,
and the request-validation explicit failures.

Checks (prefix-cache interaction checks live in
test_steering_prefix_cache.py / test_server_prefix_cache.py):
  P0  request validation: triggerless and moe-no-target configs rejected
  P2  1-token prompt: prefill [-1] steers exactly abs position 0
  P3  chunked prompt (64,64,1): prefill [-1] covers every prompt token,
      including the 1-token tail chunk
  P4  chunked prompt: generate [-1] steers only decode tokens (the
      1-token tail chunk is NOT steered as decode)
  P5  chunked prompt: prefill_trigger_positions=[-1] fires exactly once,
      at the true last prompt token
  P6  1-token prompt: generate [-1] steers only decode tokens
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
TRACE_DIR = tempfile.mkdtemp(prefix="steer_trace_chunk_")
os.environ["VLLM_STEER_TRACE_DIR"] = TRACE_DIR

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.steer_vectors.request import SteerVectorRequest  # noqa: E402

MODEL = os.environ.get(
    "STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"
)
VECTOR = os.environ.get(
    "STEER_TEST_VECTOR",
    os.path.expanduser("~/EasySteer/vectors/happy_diffmean.gguf"),
)
LAYER = 10
PROMPT_LONG = list(range(100, 100 + 129))  # chunks of 64, 64, 1
PROMPT_ONE = [100]

FAILURES = []
_next_id = [1]


def check(name, cond, detail=""):
    if cond:
        print(f"OK: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        FAILURES.append(name)


def expect_raises(name, exc_type, fn, needle=""):
    try:
        fn()
    except exc_type as e:
        check(name, needle in str(e), f"message: {e}")
    else:
        check(name, False, "did not raise")


def make_request(**kwargs):
    _next_id[0] += 1
    return SteerVectorRequest(
        steer_vector_name=f"t{_next_id[0]}",
        steer_vector_int_id=_next_id[0],
        steer_vector_local_path=VECTOR,
        scale=0.5,
        target_layers=[LAYER],
        **kwargs,
    )


def read_trace(min_step):
    """Steps and layer-LAYER applies with trace step id > min_step."""
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


def max_step():
    steps, _ = read_trace(0)
    return max(steps, default=0)


def steered_abs_positions(steps, applies):
    """(abs_position, was_prefill) per steered position; single-request
    batches only."""
    out = []
    for rec in applies:
        step = steps[rec["step"]]
        assert len(step["req_ids"]) == 1, "test expects single-request batches"
        num_computed = step["num_computed"][0]
        is_prefill_step = step["num_output"][0] == 0
        for pos in rec["positions"]:
            out.append((pos + num_computed, is_prefill_step))
    return out


# --- P0: request validation (no engine needed) -------------------------
expect_raises(
    "P0a triggerless request rejected",
    ValueError,
    lambda: make_request(),
    needle="no trigger fields",
)
expect_raises(
    "P0b moe_router no-file without target_layers rejected",
    ValueError,
    lambda: SteerVectorRequest(
        steer_vector_name="m",
        steer_vector_int_id=99,
        algorithm="moe_router",
        moe_expert_ids=[1, 2],
        prefill_trigger_tokens=[-1],
        generate_trigger_tokens=[-1],
    ),
    needle="target_layers",
)
r = SteerVectorRequest(
    steer_vector_name="m2",
    steer_vector_int_id=98,
    algorithm="moe_router",
    moe_expert_ids=[1, 2],
    target_layers=[3],
    prefill_trigger_tokens=[-1],
    generate_trigger_tokens=[-1],
)
check("P0c valid no-file moe_router request accepted", r is not None)

# --- engine with chunked prefill enabled ------------------------------
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


def run(prompt_ids, request):
    start = max_step()
    llm.generate(
        TokensPrompt(prompt_token_ids=list(prompt_ids)),
        params,
        steer_vector_request=request,
        use_tqdm=False,
    )
    steps, applies = read_trace(start)
    return steps, steered_abs_positions(steps, applies)


# --- P2: 1-token prompt, prefill-only steering ------------------------
steps, pos = run(PROMPT_ONE, make_request(prefill_trigger_tokens=[-1]))
abs_set = {p for p, _ in pos}
check(
    "P2 1-token prompt steered as prefill at abs 0",
    abs_set == {0} and all(is_pf for _, is_pf in pos),
    f"got {sorted(abs_set)}",
)

# --- P3: chunked prompt, prefill-only covers every prompt token -------
steps, pos = run(PROMPT_LONG, make_request(prefill_trigger_tokens=[-1]))
abs_set = {p for p, _ in pos}
prefill_steps = [s for s in steps.values() if s["num_output"][0] == 0]
tail = [
    s
    for s in prefill_steps
    if len(s["token_ids"]) == 1 and s["num_computed"][0] == 128
]
check(
    "P3a chunked prefill really chunked (64,64,1)",
    len(prefill_steps) == 3 and len(tail) == 1,
    f"prefill steps: {len(prefill_steps)}",
)
check(
    "P3b prefill [-1] covers all 129 prompt tokens incl. 1-token tail",
    abs_set == set(range(129)),
    f"missing {sorted(set(range(129)) - abs_set)[:5]}, "
    f"extra {sorted(abs_set - set(range(129)))[:5]}",
)

# --- P4: chunked prompt, generate-only steers decode tokens only ------
steps, pos = run(PROMPT_LONG, make_request(generate_trigger_tokens=[-1]))
abs_set = {p for p, _ in pos}
check(
    "P4 generate [-1] under chunking steers only decode tokens",
    abs_set == {129, 130, 131} and all(not is_pf for _, is_pf in pos),
    f"got {sorted(abs_set)}",
)

# --- P5: negative prefill position under chunking ---------------------
steps, pos = run(PROMPT_LONG, make_request(prefill_trigger_positions=[-1]))
abs_list = sorted(p for p, _ in pos)
check(
    "P5 prefill position -1 fires once at last prompt token (128)",
    abs_list == [128],
    f"got {abs_list}",
)

# --- P6: 1-token prompt, generate-only --------------------------------
steps, pos = run(PROMPT_ONE, make_request(generate_trigger_tokens=[-1]))
abs_set = {p for p, _ in pos}
check(
    "P6 1-token prompt: generate [-1] steers only decode tokens",
    abs_set == {1, 2, 3},
    f"got {sorted(abs_set)}",
)

shutil.rmtree(TRACE_DIR, ignore_errors=True)
print(f"\nOVERALL: {'FAIL' if FAILURES else 'PASS'}")
sys.exit(1 if FAILURES else 0)
