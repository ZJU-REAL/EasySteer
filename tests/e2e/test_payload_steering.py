# SPDX-License-Identifier: Apache-2.0
"""In-memory payload steering (VectorSpec.data) against the file path.

One eager dense engine, greedy sampling. Covers:
- compare a DirectionVector payload built from the test GGUF with
  source= loading of the same file;
- from_control_vector: an extraction-shaped object steers with no file.

Payload fingerprint and kind validation live in cpu/test_payloads.py.

async_scheduling is pinned off to keep request scheduling consistent.
"""

import numpy as np
import pytest

from vllm import SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

from helpers import DENSE_MODEL, DENSE_VECTOR, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_algorithms=["direct"],
    enforce_eager=True,
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.18,
    max_model_len=2048,
    async_scheduling=False,
)

TEXT = "The capital of France is"
SP = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
LAYERS = list(range(10, 26))
APPLY = ApplySpec(prompt="all", generation="all")


def data_spec(payload, scale=0.5):
    return SteeringSpec(
        vectors=[
            VectorSpec(data=payload, scale=scale, layers=LAYERS, apply=APPLY)
        ]
    )


def gen(llm, steering):
    out = llm.generate(TEXT, sampling_params=SP, steering=steering, use_tqdm=False)
    return out[0].outputs[0].text


@pytest.fixture(scope="module")
def plain(llm):
    return gen(llm, None)


def test_data_matches_source_byte_identical(llm, plain):
    import easysteer.vectors as vec

    payload = vec.from_gguf(DENSE_VECTOR)
    via_source = gen(llm, steering_spec(scale=2.0, layers=tuple(LAYERS)))
    via_data = gen(llm, data_spec(payload, scale=2.0))
    assert via_data == via_source, (
        "data= must reproduce source= byte-for-byte for the same vector"
    )
    assert via_data != plain, "the payload must actually steer"


def test_from_control_vector_no_disk(llm, plain):
    import easysteer.vectors as vec

    class FakeCV:
        directions = {
            la: np.ones(1536, dtype=np.float32) * 0.02 for la in LAYERS
        }

    payload = vec.from_control_vector(FakeCV())
    steered = gen(llm, data_spec(payload, scale=8.0))
    assert steered != plain
