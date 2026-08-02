#!/usr/bin/env python3
"""Server-level steering with prefix caching (salt + reset pattern).

Mechanism-level checks via RequestOutput.num_cached_tokens:

  R1  engine with server steering + prefix caching boots (ban lifted)
  R2  same prompt twice: cache reused under server steering
  R3  scale update (worker RPC + reset_prefix_cache, mirroring the
      POST /v1/steering endpoint): next request is cold, then warm again
"""

import os
import sys

GPU_ID = os.environ.get("GPU_ID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL = os.environ.get(
    "STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"
)
VECTOR = os.environ.get(
    "STEER_TEST_VECTOR",
    os.path.expanduser("~/EasySteer/vectors/happy_diffmean.gguf"),
)
PROMPT = list(range(400, 448))
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"OK: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        FAILURES.append(name)


llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    steer_vector_path=VECTOR,
    steer_scale=1.0,
    steer_target_layers=[10],
    steer_normalize=False,
    enable_prefix_caching=True,
    enable_chunked_prefill=False,
    enforce_eager=True,
    gpu_memory_utilization=0.18,
    max_model_len=512,
)
check("R1 server steering + prefix caching boots", True)
params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)


def run():
    return llm.generate(
        TokensPrompt(prompt_token_ids=list(PROMPT)), params, use_tqdm=False
    )[0]


out = run()
check("R2a cold: no cache hit", out.num_cached_tokens == 0,
      f"cached={out.num_cached_tokens}")
out = run()
check("R2b warm: cache reused under server steering",
      out.num_cached_tokens > 0, f"cached={out.num_cached_tokens}")

# --- R3: scale update, mirroring POST /v1/steering ---------------------
steer_config = llm.llm_engine.vllm_config.steer_vector_config
object.__setattr__(steer_config, "server_scale", 2.0)
from vllm.steer_vectors.request import build_server_request  # noqa: E402

llm.llm_engine.collective_rpc(
    "add_steer_vector", args=(build_server_request(steer_config),)
)
llm.reset_prefix_cache()

out = run()
check("R3a after scale update + reset: cold",
      out.num_cached_tokens == 0, f"cached={out.num_cached_tokens}")
out = run()
check("R3b new-scale blocks reused",
      out.num_cached_tokens > 0, f"cached={out.num_cached_tokens}")

print(f"\nOVERALL: {'FAIL' if FAILURES else 'PASS'}")
sys.exit(1 if FAILURES else 0)
