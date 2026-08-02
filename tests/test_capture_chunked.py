#!/usr/bin/env python3
"""Capture coverage and reductions under chunked prefill.

  K1  positions='all': every prompt token captured exactly once across
      chunks (129-token prompt, 64-token budget -> chunks 64/64/1),
      plus one row per decode step
  K2  positions='last': one row per logical step (final prompt chunk +
      each decode step), not one per chunk
  K3  1-token prompt with 'last': same semantics
"""

import os
import sys

GPU_ID = os.environ.get("GPU_ID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.hidden_states import deserialize_hidden_states  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL = os.environ.get(
    "STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"
)
PROMPT_LONG = list(range(100, 229))  # 129 tokens
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"OK: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        FAILURES.append(name)


llm = LLM(
    model=MODEL,
    enforce_eager=True,
    enable_prefix_caching=False,
    enable_chunked_prefill=True,
    max_num_batched_tokens=64,
    max_num_seqs=4,
    max_model_len=512,
    gpu_memory_utilization=0.18,
)
params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)


def rpc(method, *args, **kwargs):
    return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)[0]


def capture_rows(prompt_ids, positions):
    rpc("start_capture", "hidden_states", layers=[0], positions=positions)
    llm.generate(
        TokensPrompt(prompt_token_ids=list(prompt_ids)), params, use_tqdm=False
    )
    hs = deserialize_hidden_states(rpc("fetch_captured", "hidden_states"))
    rpc("stop_capture", "hidden_states")
    return hs[0].shape[0]


# K1: 'all' coverage across chunks: 129 prompt + 3 decode forwards
rows = capture_rows(PROMPT_LONG, "all")
check("K1 'all' covers all chunks + decode", rows == 132, f"rows={rows}")

# K2: 'last' is per logical step, not per chunk: 1 final-chunk + 3 decode
rows = capture_rows(PROMPT_LONG, "last")
check("K2 'last' one row per step under chunking", rows == 4, f"rows={rows}")

# K3: 1-token prompt with 'last': 1 prefill + 3 decode
rows = capture_rows([100], "last")
check("K3 1-token prompt 'last'", rows == 4, f"rows={rows}")

print(f"\nOVERALL: {'FAIL' if FAILURES else 'PASS'}")
sys.exit(1 if FAILURES else 0)
