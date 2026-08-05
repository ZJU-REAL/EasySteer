# SPDX-License-Identifier: Apache-2.0
"""max_steer_vectors backpressure on the in-graph tier.

Regression for the capacity-overflow crash: a capacity-2 in_graph
engine given more concurrent distinct configurations than slots used
to die with "No free steering rows" (EngineDeadError) because the
scheduler counted distinct vector *files* while the worker allocates
by config *fingerprint*. With fingerprint-keyed scheduling the excess
requests wait for a slot instead: everything completes and the engine
stays alive.
"""

from vllm import SamplingParams

from helpers import DENSE_MODEL, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_algorithms=["direct"],
    max_steer_vectors=2,
    steer_graph_mode="in_graph",
    enforce_eager=False,
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.25,
    max_model_len=2048,
    max_num_seqs=16,
)

PROMPT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
LAYERS = list(range(10, 26))


def gen(llm, prompts, steering=None, max_tokens=16):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                        ignore_eos=True)
    return llm.generate(prompts, steering=steering, sampling_params=sp,
                        use_tqdm=False)


def test_overflow_no_longer_kills_the_engine(llm):
    """Six distinct same-file configs on two slots: all requests
    complete (the old behavior was EngineDeadError on the third)."""
    steering = [
        steering_spec(scale=0.0, layers=LAYERS, exclude_positions=[i])
        for i in range(6)
    ]
    outs = gen(llm, [PROMPT] * 6, steering)
    assert all(len(o.outputs[0].token_ids) == 16 for o in outs)


def test_engine_alive_and_steering_after_overflow(llm):
    """The engine still serves — and steers — after the overflow
    workload."""
    plain = gen(llm, [PROMPT])[0].outputs[0].text
    steered = gen(
        llm, [PROMPT], steering=steering_spec(scale=2.0, layers=LAYERS)
    )[0].outputs[0].text
    assert steered != plain, "steering inert after overflow workload"
