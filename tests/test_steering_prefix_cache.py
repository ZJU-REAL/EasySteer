#!/usr/bin/env python3
"""Steering-aware prefix caching: KV reuse keyed by config fingerprint.

Mechanism-level checks via RequestOutput.num_cached_tokens (no reliance
on output text, so no batch-shape numerics sensitivity):

  S1  same prompt + same config: cache reused
  S2  same prompt + different scale: NO reuse (the 9b999cb collision)
  S3  the different-scale config then reuses its own blocks
  S4  unsteered requests never reuse steered blocks (and vice versa),
      but unsteered<->unsteered reuse works
  S5  length-sensitive config (negative position): no reuse across
      different prompt lengths, reuse at equal length
  S6  prompt containing another request's generated tokens reuses only
      the prompt-region blocks (phase boundary respected)
  S7  capture cannot be enabled on a prefix-caching engine
  S8  fresh runtime install of server-level steering rejected on a
      prefix-caching engine (pre-install blocks are unsalted)
"""

import os
import sys

GPU_ID = os.environ.get("GPU_ID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

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
PROMPT_48 = list(range(200, 248))
FAILURES = []
_next_id = [10]


def check(name, cond, detail=""):
    if cond:
        print(f"OK: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        FAILURES.append(name)


def make_request(**kwargs):
    _next_id[0] += 1
    kwargs.setdefault("prefill_trigger_tokens", [-1])
    kwargs.setdefault("generate_trigger_tokens", [-1])
    return SteerVectorRequest(
        steer_vector_name=f"pc{_next_id[0]}",
        steer_vector_int_id=_next_id[0],
        steer_vector_local_path=VECTOR,
        target_layers=[10],
        **kwargs,
    )


llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    enable_prefix_caching=True,  # allowed with per-request steering
    enable_chunked_prefill=False,
    enforce_eager=True,
    gpu_memory_utilization=0.18,
    max_model_len=512,
)
params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)


def run(prompt_ids, request=None, max_tokens=4):
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    out = llm.generate(
        TokensPrompt(prompt_token_ids=list(prompt_ids)),
        sp,
        steer_vector_request=request,
        use_tqdm=False,
    )[0]
    return out


cfg_a = make_request(scale=1.0)
cfg_b = make_request(scale=3.0)

# --- S1/S2/S3: fingerprint keying --------------------------------------
out = run(PROMPT_48, cfg_a)
check("S1a config A cold: no cache hit", out.num_cached_tokens == 0,
      f"cached={out.num_cached_tokens}")
out = run(PROMPT_48, cfg_a)
check("S1b config A warm: cache reused", out.num_cached_tokens > 0,
      f"cached={out.num_cached_tokens}")
out = run(PROMPT_48, cfg_b)
check("S2 different scale: NO reuse", out.num_cached_tokens == 0,
      f"cached={out.num_cached_tokens}")
out = run(PROMPT_48, cfg_b)
check("S3 same different-scale config: reuses its own blocks",
      out.num_cached_tokens > 0, f"cached={out.num_cached_tokens}")

# --- S4: steered vs unsteered isolation --------------------------------
out = run(PROMPT_48, None)
check("S4a unsteered does not reuse steered blocks",
      out.num_cached_tokens == 0, f"cached={out.num_cached_tokens}")
out = run(PROMPT_48, None)
check("S4b unsteered reuses unsteered blocks",
      out.num_cached_tokens > 0, f"cached={out.num_cached_tokens}")

# --- S5: length-sensitive configs --------------------------------------
cfg_c = make_request(
    prefill_trigger_tokens=None,
    generate_trigger_tokens=None,
    prefill_trigger_positions=[-1],
)
prompt_56 = PROMPT_48 + list(range(300, 308))
out = run(PROMPT_48, cfg_c)
check("S5a length-sensitive cold: no hit", out.num_cached_tokens == 0,
      f"cached={out.num_cached_tokens}")
out = run(prompt_56, cfg_c)
check("S5b longer prompt, same config: NO reuse (plen key)",
      out.num_cached_tokens == 0, f"cached={out.num_cached_tokens}")
out = run(PROMPT_48, cfg_c)
check("S5c equal length, same config: reuse",
      out.num_cached_tokens > 0, f"cached={out.num_cached_tokens}")

# --- S6: phase boundary ------------------------------------------------
cfg_d = make_request(scale=0.5)
out = run(PROMPT_48, cfg_d, max_tokens=20)
generated = list(out.outputs[0].token_ids)
continuation = PROMPT_48 + generated  # 68-token prompt
out = run(continuation, cfg_d)
check(
    "S6 continuation reuses only prompt-region blocks",
    0 < out.num_cached_tokens <= 48,
    f"cached={out.num_cached_tokens} (prompt region is 48)",
)

# --- S7: capture rejected on prefix-caching engine ---------------------
try:
    llm.llm_engine.collective_rpc("enable_hidden_states_capture")
except Exception as e:
    check("S7 capture rejected under prefix caching",
          "prefix" in str(e).lower(), f"message: {e}")
else:
    check("S7 capture rejected under prefix caching", False, "did not raise")

# --- S8: fresh runtime server-steering install rejected ----------------
try:
    llm.llm_engine.collective_rpc(
        "add_steer_vector", args=(make_request(scale=1.0),)
    )
except Exception as e:
    check("S8 fresh server install rejected under prefix caching",
          "salt" in str(e).lower(), f"message: {e}")
else:
    check("S8 fresh server install rejected under prefix caching",
          False, "did not raise")

print(f"\nOVERALL: {'FAIL' if FAILURES else 'PASS'}")
sys.exit(1 if FAILURES else 0)
