# SPDX-License-Identifier: Apache-2.0
"""Validate Tier-1 full-graph steering (steer_graph_mode=full).

The steering kernel `hidden += mask * vectors[row_tok]` reads persistent
buffers and captures into full CUDA graphs; triggers/routing are computed
host-side each step. Checks:
  F1  full cudagraphs are kept (no piecewise downgrade);
  F2  steering fires (steered != unsteered, upbeat);
  F3  scale-0 steering is byte-identical to no steering;
  F4  repeated steered run is byte-identical (graph replay determinism);
  F5  routing isolation: in a [steered, plain] batch the plain request is
      byte-identical to the same batch without steering (ignore_eos keeps
      geometries aligned);
  F6  non-graph-safe config (loreft) is rejected with a clear error;
  F7  a v2 SteeringSpec drives the same full-graph kernel (backend
      independence): steered != plain, and identical to the equivalent
      v1 request's output.

Env: GPU_ID; STEER_TEST_EAGER=1 runs the same kernel path eagerly.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

MODEL = os.environ.get("STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/")
VEC = os.path.expanduser(os.environ.get("STEER_TEST_VECTOR", "~/EasySteer/vectors/happy_diffmean.gguf"))
EAGER = os.environ.get("STEER_TEST_EAGER", "0") == "1"

llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    steer_graph_mode="full",
    enforce_eager=EAGER,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.25,
    max_model_len=2048,
)

comp = llm.llm_engine.vllm_config.compilation_config
print(f"cudagraph_mode: {comp.cudagraph_mode}")

failures = []
if not EAGER and not comp.cudagraph_mode.has_full_cudagraphs():
    failures.append(f"F1: cudagraph_mode {comp.cudagraph_mode} has no "
                    "full graphs")

sp = SamplingParams(temperature=0.0, max_tokens=96, ignore_eos=True)
text = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
layers = list(range(10, 26))


def req(name, i, scale):
    return SteerVectorRequest(
        name, i, steer_vector_local_path=VEC, scale=scale,
        target_layers=layers,
        prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])


def gen(prompts, reqs):
    outs = llm.generate(prompts, steer_vector_request=reqs,
                        sampling_params=sp)
    return [o.outputs[0].text for o in outs]


plain = gen([text], None)[0]
zero = gen([text], req("zero", 1, 0.0))[0]
happy = gen([text], req("happy", 2, 2.0))[0]
happy2 = gen([text], req("happy2", 3, 2.0))[0]
batch_mixed = gen([text, text], [req("happy3", 4, 2.0), None])
batch_plain = gen([text, text], None)

if happy == plain:
    failures.append("F2: steered output identical to unsteered")
if zero != plain:
    failures.append("F3: scale-0 differs from no steering")
if happy != happy2:
    failures.append("F4: repeated steered run not deterministic")
if batch_mixed[1] != batch_plain[1]:
    failures.append("F5: plain request contaminated in mixed batch")

try:
    gen([text], SteerVectorRequest(
        "bad", 9, steer_vector_local_path=VEC, scale=1.0,
        algorithm="loreft", target_layers=layers,
        prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1]))
    failures.append("F6: loreft accepted in full-graph mode")
except Exception as e:
    print(f"F6 OK: loreft rejected ({type(e).__name__})")

from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec  # noqa: E402

v2_happy = llm.generate(
    [text],
    steering=SteeringSpec(vectors=[VectorSpec(
        source=VEC, scale=2.0, layers=layers,
        apply=ApplySpec(phases=["prompt", "generation"]),
    )]),
    sampling_params=sp,
)[0].outputs[0].text
if v2_happy == plain:
    failures.append("F7: v2 spec did not steer under full graphs")
if v2_happy != happy:
    failures.append("F7: v2 spec output differs from equivalent v1 request")

print("======unsteered======")
print(plain)
print("======happy steer (full graph)======")
print(happy)
for f in failures:
    print("FAIL:", f)
print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
