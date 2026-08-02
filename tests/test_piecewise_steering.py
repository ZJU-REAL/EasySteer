# SPDX-License-Identifier: Apache-2.0
"""Validate Tier-2 piecewise CUDA-graph steering (vllm::steer_apply).

Boots the engine WITHOUT enforce_eager so the model is torch.compiled and
piecewise cudagraphs are captured, then checks:
  C1  config wiring: VLLM_COMPILE mode, PIECEWISE cudagraphs, and
      vllm::steer_apply present in splitting_ops;
  C2  steering fires under compiled execution (steered != unsteered) —
      this is the spike proof that forward hooks trace into the graph;
  C3  scale-0 steering is byte-identical to no steering (op is a clean
      no-op on unsteered tokens);
  C4  a repeated steered run is byte-identical (cudagraph replay is
      deterministic);
  C5  informational: diff vs the eager golden (compiled kernels may
      drift numerically from eager — not a failure).

Env: GPU_ID selects the device (CUDA_DEVICE_ORDER should be PCI_BUS_ID
on mixed-GPU hosts).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

MODEL = os.environ.get("STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/")
VEC = os.path.expanduser(os.environ.get("STEER_TEST_VECTOR", "~/EasySteer/vectors/happy_diffmean.gguf"))
GOLDEN = os.path.expanduser("~/EasySteer-migration/golden.txt")

llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    enforce_eager=False,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.25,
    max_model_len=2048,
)

cfg = llm.llm_engine.vllm_config
comp = cfg.compilation_config
print(f"compilation mode: {comp.mode}, cudagraph_mode: {comp.cudagraph_mode}")
print(f"splitting_ops has steer_apply: "
      f"{'vllm::steer_apply' in (comp.splitting_ops or [])}")

failures = []
if "vllm::steer_apply" not in (comp.splitting_ops or []):
    failures.append("C1: vllm::steer_apply missing from splitting_ops")
if not comp.cudagraph_mode.has_piecewise_cudagraphs():
    failures.append(f"C1: cudagraph_mode is {comp.cudagraph_mode}, "
                    "expected piecewise")

sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
text = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
target_layers = list(range(10, 26))


def gen(req):
    out = llm.generate(text, steer_vector_request=req,
                       sampling_params=sampling_params)
    return out[0].outputs[0].text


plain = llm.generate(text, sampling_params=sampling_params)[0].outputs[0].text
zero = gen(SteerVectorRequest(
    "baseline", 1, steer_vector_local_path=VEC, scale=0,
    target_layers=target_layers,
    prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1]))
happy = gen(SteerVectorRequest(
    "happy", 2, steer_vector_local_path=VEC, scale=2.0,
    target_layers=target_layers,
    prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1]))
happy2 = gen(SteerVectorRequest(
    "happy-again", 3, steer_vector_local_path=VEC, scale=2.0,
    target_layers=target_layers,
    prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1]))

if happy == zero:
    failures.append("C2: steered output identical to unsteered "
                    "(steering NOT firing under compiled execution)")
if zero != plain:
    failures.append("C3: scale-0 steering differs from no steering")
if happy != happy2:
    failures.append("C4: repeated steered run not deterministic")

print("======unsteered (compiled)======")
print(plain)
print("======happy steer (compiled)======")
print(happy)

if os.path.exists(GOLDEN):
    golden = open(GOLDEN).read()
    print("C5 (info): compiled outputs "
          + ("MATCH" if (zero in golden and happy in golden) else "DIFFER FROM")
          + " eager golden (numeric drift from compiled kernels is expected)")

for f in failures:
    print("FAIL:", f)
print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
