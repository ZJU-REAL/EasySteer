# SPDX-License-Identifier: Apache-2.0
"""Steering-enabled engines must not perturb unsteered traffic.

Steering is row-local on the hidden stream (no residual collapse), so an
engine with enable_steer_vector=True and no steering config must produce
byte-identical outputs to a vanilla engine. Boots both engines in one
process at low memory utilization (they fit together even if the first
does not release; the engine-per-module convention guards GPU state
release, which this comparison does not depend on).
"""

import gc

from vllm import LLM, SamplingParams

from helpers import DENSE_MODEL

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "In machine learning, overfitting means",
    "The fastest land animal is",
]

COMMON = dict(
    model=DENSE_MODEL,
    enforce_eager=True,
    gpu_memory_utilization=0.18,
    max_model_len=2048,
    async_scheduling=False,
)


def _greedy_token_ids(llm):
    sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
    outs = llm.generate(PROMPTS, sampling_params=sp)
    return [tuple(o.outputs[0].token_ids) for o in outs]


def test_unsteered_traffic_matches_vanilla():
    vanilla = LLM(**COMMON)
    want = _greedy_token_ids(vanilla)
    del vanilla
    gc.collect()

    steering_enabled = LLM(enable_steer_vector=True, **COMMON)
    got = _greedy_token_ids(steering_enabled)
    assert got == want, (
        "enable_steer_vector=True changed unsteered outputs relative to a "
        "vanilla engine"
    )
