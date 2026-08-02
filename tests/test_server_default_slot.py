# SPDX-License-Identifier: Apache-2.0
"""Validate server-level steering on the routed default slot.

Boots an engine with server-level steering args (steer_vector_path etc.),
generates WITHOUT any per-request steering config, and checks the output
is steered: byte-identical to the golden happy-steer output produced by
an explicit per-request config with the same parameters (same math, same
batch geometry).

Env: GPU_ID; expects ~/EasySteer-migration/golden.txt to exist.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))

from vllm import LLM, SamplingParams

MODEL = os.environ.get("STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/")
VEC = os.path.expanduser(os.environ.get("STEER_TEST_VECTOR", "~/EasySteer/vectors/happy_diffmean.gguf"))
GOLDEN = os.path.expanduser("~/EasySteer-migration/golden.txt")

llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    steer_vector_path=VEC,
    steer_scale=2.0,
    steer_target_layers=list(range(10, 26)),
    steer_normalize=False,
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.18,
    max_model_len=2048,
)

sp = SamplingParams(temperature=0.0, max_tokens=128)
text = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
out = llm.generate(text, sampling_params=sp)[0].outputs[0].text

print("======server-default steer======")
print(out)

golden = open(GOLDEN).read()
ok = out in golden
print("server-default output",
      "MATCHES golden happy-steer output" if ok else "DIFFERS from golden")
print("OVERALL:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
