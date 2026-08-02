# SPDX-License-Identifier: Apache-2.0
"""Validate --steer-require-preload frontend enforcement.

With require_preload set:
  P1  a steering request whose vector was NOT preloaded is rejected with
      a clear ValueError at the frontend (engine stays alive);
  P2  after LLM.preload_steer_vectors, the same request succeeds and
      actually steers;
  P3  an unpreloaded multi-vector request is rejected too.

Env: GPU_ID.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig

MODEL = os.environ.get("STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/")
VEC = os.path.expanduser(os.environ.get("STEER_TEST_VECTOR", "~/EasySteer/vectors/happy_diffmean.gguf"))

llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    steer_require_preload=True,
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.18,
    max_model_len=2048,
)

sp = SamplingParams(temperature=0.0, max_tokens=48)
text = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
layers = list(range(10, 26))
req = SteerVectorRequest(
    "happy", 1, steer_vector_local_path=VEC, scale=2.0,
    target_layers=layers,
    prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])

failures = []

try:
    llm.generate(text, steer_vector_request=req, sampling_params=sp)
    failures.append("P1: unpreloaded vector was accepted")
except ValueError as e:
    if "not preloaded" not in str(e):
        failures.append(f"P1: wrong error message: {e}")
    else:
        print("P1 OK: rejected with:", str(e)[:100])

plain = llm.generate(text, sampling_params=sp)[0].outputs[0].text

llm.preload_steer_vectors([VEC])
steered = llm.generate(
    text, steer_vector_request=req, sampling_params=sp
)[0].outputs[0].text
if steered == plain:
    failures.append("P2: preloaded request did not steer")
else:
    print("P2 OK: steering active after preload")

multi_req = SteerVectorRequest(
    "multi", 2, vector_configs=[VectorConfig(
        path=VEC + ".does-not-exist", scale=1.0, target_layers=layers,
        prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])])
try:
    llm.generate(text, steer_vector_request=multi_req, sampling_params=sp)
    failures.append("P3: unpreloaded multi-vector accepted")
except ValueError as e:
    if "not preloaded" not in str(e):
        failures.append(f"P3: wrong error message: {e}")
    else:
        print("P3 OK: multi-vector rejected")

for f in failures:
    print("FAIL:", f)
print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
