# SPDX-License-Identifier: Apache-2.0
"""Validate the hook-based capture rework on a dense model (V2 runner).

Checks:
  H1  legacy RPC flow captures all 28 layers, row counts consistent and
      equal to prefill + decode tokens;
  H2  layer subset config captures only those layers;
  H3  positions='last' stores one row per sample per step;
  H4  positions='mean' same row count, different values;
  H5  max_tokens caps rows per layer and reports drops;
  H6  dtype='float16' round-trips through serialization;
  H7  router_logits stream on a dense model yields nothing (with a
      warning, not an error);
  H8  fetch(clear=True) resets accumulation.

Env: GPU_ID.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))

import torch
from vllm import LLM, SamplingParams
from vllm.hidden_states import deserialize_hidden_states

MODEL = os.environ.get("STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/")
NUM_LAYERS = 28

llm = LLM(
    model=MODEL,
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.18,
    max_model_len=2048,
)


def rpc(method, *args, **kwargs):
    return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)[0]


sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
text = "The capital of France is"
prompt_tokens = len(llm.get_tokenizer().encode(text))
expected_rows = prompt_tokens + 7  # prefill + 7 decode steps
failures = []


def check(name, cond, detail=""):
    if not cond:
        failures.append(f"{name} {detail}")
        print("FAIL:", name, detail)
    else:
        print("OK:", name)


# H1: legacy flow
rpc("enable_hidden_states_capture")
llm.generate(text, sampling_params=sp)
hs = deserialize_hidden_states(rpc("get_captured_hidden_states"))
rows = {t.shape[0] for t in hs.values()}
check("H1 all layers", len(hs) == NUM_LAYERS, f"got {len(hs)}")
check("H1 row counts", rows == {expected_rows},
      f"rows={rows} expected={expected_rows}")
rpc("clear_hidden_states")
rpc("disable_hidden_states_capture")

# H2: layer subset
rpc("start_capture", "hidden_states", layers=[5, 10])
llm.generate(text, sampling_params=sp)
d = rpc("fetch_captured", "hidden_states")
check("H2 layer subset", sorted(d.keys()) == [5, 10],
      f"got {sorted(d.keys())}")
rpc("stop_capture", "hidden_states")

# H3/H4: per-sample reductions
rpc("start_capture", "hidden_states", layers=[10], positions="last")
llm.generate(text, sampling_params=sp)
last = deserialize_hidden_states(rpc("fetch_captured", "hidden_states"))
check("H3 positions=last", last[10].shape[0] == 8,
      f"rows={last[10].shape[0]}")
rpc("stop_capture", "hidden_states")

rpc("start_capture", "hidden_states", layers=[10], positions="mean")
llm.generate(text, sampling_params=sp)
mean = deserialize_hidden_states(rpc("fetch_captured", "hidden_states"))
check("H4 positions=mean rows", mean[10].shape[0] == 8,
      f"rows={mean[10].shape[0]}")
check("H4 mean != last",
      not torch.allclose(mean[10][0].float(), last[10][0].float()))
rpc("stop_capture", "hidden_states")

# H5: per-layer token budget
rpc("start_capture", "hidden_states", layers=[10], max_tokens=5)
llm.generate(text, sampling_params=sp)
status = rpc("capture_status", "hidden_states")
capped = deserialize_hidden_states(rpc("fetch_captured", "hidden_states"))
check("H5 budget cap", capped[10].shape[0] == 5,
      f"rows={capped[10].shape[0]}")
check("H5 drops reported", status["tokens_dropped"] > 0, str(status))
rpc("stop_capture", "hidden_states")

# H6: dtype
rpc("start_capture", "hidden_states", layers=[10], dtype="float16")
llm.generate(text, sampling_params=sp)
f16 = deserialize_hidden_states(rpc("fetch_captured", "hidden_states"))
check("H6 dtype", f16[10].dtype == torch.float16, str(f16[10].dtype))
rpc("stop_capture", "hidden_states")

# H7: router logits on a dense model
rpc("start_capture", "router_logits")
llm.generate(text, sampling_params=sp)
check("H7 dense router empty",
      rpc("fetch_captured", "router_logits") == {})
rpc("stop_capture", "router_logits")

# H8: fetch clears
rpc("start_capture", "hidden_states", layers=[10])
llm.generate(text, sampling_params=sp)
first = deserialize_hidden_states(rpc("fetch_captured", "hidden_states"))
llm.generate(text, sampling_params=sp)
second = deserialize_hidden_states(rpc("fetch_captured", "hidden_states"))
check("H8 fetch clears",
      first[10].shape[0] == expected_rows
      and second[10].shape[0] == expected_rows,
      f"{first[10].shape[0]}/{second[10].shape[0]}")

for f in failures:
    print("FAIL:", f)
print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
