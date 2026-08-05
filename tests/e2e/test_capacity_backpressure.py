# SPDX-License-Identifier: Apache-2.0
"""max_steer_vectors as a scheduling constraint (split/eager tier).

Concurrent distinct configurations are bounded like max_loras: the
scheduler defers waiting requests whose config fingerprint cannot get
a slot, and admits them as running configs release theirs. One eager
capacity-2 engine; the steering trace proves the bound — no scheduler
step ever contains more than two distinct config slots — while every
request still completes and steers correctly.
"""

import os

from vllm import SamplingParams

from helpers import DENSE_MODEL, read_trace, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_algorithms=["direct"],
    max_steer_vectors=2,
    enforce_eager=True,
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.3,
    max_model_len=2048,
    max_num_seqs=16,
    async_scheduling=False,
)

PROMPT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
LAYERS = list(range(10, 26))
CAPACITY = 2


def distinct_spec(i, scale=0.0):
    return steering_spec(scale=scale, layers=LAYERS, exclude_positions=[i])


def gen(llm, prompts, steering, max_tokens=32):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                        ignore_eos=True)
    outs = llm.generate(prompts, steering=steering, sampling_params=sp,
                        use_tqdm=False)
    return outs


def test_overflow_completes_and_respects_capacity(llm):
    """8 requests with 6 distinct configs on a capacity-2 engine: all
    complete, and no scheduler step carries more than 2 distinct
    slots."""
    trace_dir = os.environ["VLLM_STEER_TRACE_DIR"]
    steps_before, _ = read_trace(trace_dir, 0, ())
    start = max(steps_before, default=0)

    steering = [distinct_spec(i) for i in range(6)] + [None, None]
    outs = gen(llm, [PROMPT] * 8, steering)
    assert all(len(o.outputs[0].token_ids) == 32 for o in outs), (
        "some request did not run to completion under slot backpressure"
    )

    steps, _ = read_trace(trace_dir, start, ())
    assert steps, "no scheduler steps recorded"
    for step_id, step in steps.items():
        distinct = {s for s in step["slots"] if s >= 0}
        assert len(distinct) <= CAPACITY, (
            f"step {step_id} carried {len(distinct)} distinct config "
            f"slots (> capacity {CAPACITY}): {sorted(distinct)}"
        )


def test_steering_still_fires_under_backpressure(llm):
    """A real-scale config queued behind capacity still steers."""
    plain = gen(llm, [PROMPT], None)[0].outputs[0].text
    steering = [distinct_spec(i) for i in range(4)] + [
        steering_spec(scale=2.0, layers=LAYERS)
    ]
    outs = gen(llm, [PROMPT] * 5, steering)
    assert outs[4].outputs[0].text != plain, (
        "the throttled real-scale config did not steer"
    )
