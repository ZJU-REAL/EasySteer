# SPDX-License-Identifier: Apache-2.0
"""--steer-require-preload frontend enforcement.

With require_preload set, steering specs referencing vectors that were
not explicitly preloaded are rejected with a clear ValueError at the
frontend — the engine stays alive — and succeed (and actually steer)
after LLM.preload_steer_vectors.
"""

import pytest

from vllm import SamplingParams

from helpers import DENSE_MODEL, DENSE_VECTOR, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_require_preload=True,
    enforce_eager=True,
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.18,
    max_model_len=2048,
)

PROMPT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
LAYERS = list(range(10, 26))
SP = SamplingParams(temperature=0.0, max_tokens=48)


def test_unpreloaded_vector_rejected_then_accepted(llm):
    spec = steering_spec(scale=2.0, layers=LAYERS)

    with pytest.raises(ValueError, match="not preloaded"):
        llm.generate(PROMPT, steering=spec, sampling_params=SP)

    plain = llm.generate(PROMPT, sampling_params=SP)[0].outputs[0].text

    llm.preload_steer_vectors([DENSE_VECTOR])
    steered = llm.generate(
        PROMPT, steering=spec, sampling_params=SP
    )[0].outputs[0].text
    assert steered != plain, "preloaded spec must actually steer"


def test_unpreloaded_multi_vector_rejected(llm):
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    missing = SteeringSpec(vectors=[VectorSpec(
        source=DENSE_VECTOR.replace(".gguf", "-does-not-exist.gguf"),
        layers=LAYERS,
        apply=ApplySpec(phases=["prompt", "generation"]),
    ), VectorSpec(
        source=DENSE_VECTOR,
        layers=LAYERS,
        apply=ApplySpec(phases=["prompt", "generation"]),
    )])
    with pytest.raises(ValueError, match="not preloaded"):
        llm.generate(PROMPT, steering=missing, sampling_params=SP)
