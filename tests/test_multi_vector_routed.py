# SPDX-License-Identifier: Apache-2.0
"""Validate slot-routed multi-vector steering (legacy activation retired).

Checks (single engine, greedy):
  M1  a multi-vector request with ONE sub-vector is byte-identical to the
      equivalent single-vector request (same math through the multi path);
  M2  a two-sub-vector request (split target layers, sequential) steers
      (differs from unsteered);
  M3  no cross-request contamination: in a [multi, plain] batch the plain
      request's output is byte-identical to the same batch without any
      steering.

Env: GPU_ID, STEER_TEST_EAGER=0 to run compiled+piecewise.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig

MODEL = os.environ.get("STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/")
VEC = os.path.expanduser(os.environ.get("STEER_TEST_VECTOR", "~/EasySteer/vectors/happy_diffmean.gguf"))
EAGER = os.environ.get("STEER_TEST_EAGER", "1") == "1"

llm = LLM(
    model=MODEL,
    enable_steer_vector=True,
    enforce_eager=EAGER,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.25,
    max_model_len=2048,
    # M3 byte-compares one request's output across two generate calls;
    # async scheduling makes prefill co-batching timing-dependent, which
    # can change batch geometry between the runs (numeric drift, flaky
    # false "contamination").
    async_scheduling=False,
)

# ignore_eos keeps batch geometry identical between the mixed and plain
# batches (M3 byte-compares across runs; early EOS in the steered request
# would change the co-batched request's batch shapes -> numeric drift).
sp = SamplingParams(temperature=0.0, max_tokens=96, ignore_eos=True)
text = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
layers = list(range(10, 26))


def gen(prompts, reqs):
    outs = llm.generate(prompts, steer_vector_request=reqs, sampling_params=sp)
    return [o.outputs[0].text for o in outs]


single_req = SteerVectorRequest(
    "single", 1, steer_vector_local_path=VEC, scale=2.0,
    target_layers=layers,
    prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
multi_one_req = SteerVectorRequest(
    "multi-one", 2,
    vector_configs=[VectorConfig(
        path=VEC, scale=2.0, target_layers=layers,
        prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])])
multi_two_req = SteerVectorRequest(
    "multi-two", 3, conflict_resolution="sequential",
    vector_configs=[
        VectorConfig(path=VEC, scale=1.2, target_layers=list(range(10, 18)),
                     prefill_trigger_tokens=[-1],
                     generate_trigger_tokens=[-1]),
        VectorConfig(path=VEC, scale=1.2, target_layers=list(range(18, 26)),
                     prefill_trigger_tokens=[-1],
                     generate_trigger_tokens=[-1]),
    ])

plain = gen([text], None)[0]
single = gen([text], single_req)[0]
multi_one = gen([text], multi_one_req)[0]
multi_two = gen([text], multi_two_req)[0]
batch_mixed = gen([text, text], [multi_two_req, None])
batch_plain = gen([text, text], None)

failures = []
if multi_one != single:
    failures.append("M1: multi(single sub-vector) != equivalent single-vector")
if multi_two == plain:
    failures.append("M2: two-sub-vector request did not steer")
if batch_mixed[1] != batch_plain[1]:
    failures.append("M3: plain request contaminated by batched multi-vector")

print("======plain======")
print(plain)
print("======multi-two======")
print(multi_two)
for f in failures:
    print("FAIL:", f)
print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
