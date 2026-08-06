# SPDX-License-Identifier: Apache-2.0
"""In-graph steering at large slot capacity (per-token gather kernel).

Above ``max_steer_vectors > 2 * steer_graph_max_rank`` the low-rank
family switches from dense all-slot coefficients to per-token weight
gathers (the dense [slots, tokens, hidden] product grows linearly with
capacity and OOMs Inductor autotuning at a few hundred slots). This
module boots the same engine as test_fullgraph.py at capacity 96 so
every behavioral check runs through the gather formulation: the
low-rank family steers, zero scale stays bit-exact, and co-batched
plain requests stay uncontaminated.
"""

import os

from vllm import SamplingParams

from helpers import DENSE_MODEL, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_algorithms=["direct", "lm_steer", "loreft"],
    steer_graph_mode="in_graph",
    max_steer_vectors=96,  # > 2 * steer_graph_max_rank -> gather path
    enforce_eager=False,
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.25,
    max_model_len=2048,
)

TEXT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
LAYERS = list(range(10, 26))
SP = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
HIDDEN = 1536  # Qwen2.5-1.5B

LOREFT_WEIGHT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "replications", "loreft", "weight",
)


def _data_spec(payload, algorithm, scale, layers=None, **apply_kwargs):
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    if not apply_kwargs:
        apply_kwargs = {"prompt": "all", "generation": "all"}
    return SteeringSpec(vectors=[VectorSpec(
        data=payload, algorithm=algorithm, scale=scale, layers=layers,
        apply=ApplySpec(**apply_kwargs),
    )])


def gen(llm, prompts, **kwargs):
    outs = llm.generate(prompts, sampling_params=SP, use_tqdm=False, **kwargs)
    return [o.outputs[0].text for o in outs]


def test_lowrank_gather_path_steers_and_zero_is_exact(llm):
    """lm_steer (lowrank family) through the gather kernel: a rank-4
    projector at high scale changes the output; scale 0 is bit-exact."""
    import numpy as np

    from vllm.steer_vectors.payloads import LowRankProjector

    plain = gen(llm, [TEXT])[0]
    axes = np.zeros((HIDDEN, 4), dtype=np.float32)
    axes[:4, :4] = np.eye(4, dtype=np.float32)
    proj = LowRankProjector(axes, axes)
    steered = gen(llm, [TEXT],
                  steering=_data_spec(proj, "lm_steer", 50.0, LAYERS))[0]
    assert steered != plain, "lm_steer did not steer via the gather path"
    zero = gen(llm, [TEXT],
               steering=_data_spec(proj, "lm_steer", 0.0, LAYERS))[0]
    assert zero == plain, "zero-scale lm_steer is not a no-op (gather path)"


def test_loreft_emoji_gather_path(llm):
    """The replication LoReFT checkpoint behaves identically through
    the gather kernel."""
    from easysteer.vectors import from_pyreft

    prompt = "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n"
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    spec = _data_spec(from_pyreft(LOREFT_WEIGHT), "loreft", 1.0,
                      layers=[8], prompt_positions=[-1])
    plain = llm.generate(prompt, sampling_params=sp,
                         use_tqdm=False)[0].outputs[0].text
    steered = llm.generate(prompt, sampling_params=sp, use_tqdm=False,
                           steering=spec)[0].outputs[0].text
    assert steered != plain, "loreft did not change the output"
    assert any(ord(ch) >= 0x1F300 for ch in steered), (
        f"loreft output carries no emoji: {steered!r}"
    )


def test_many_distinct_configs_isolated(llm):
    """A mixed batch of distinct direct configs plus a plain request on
    the large-capacity engine: every request must match a plain-only
    run byte-exactly (additive family + row routing at capacity 96).

    Same nondeterminism guard as test_fullgraph's mixed-batch oracle:
    on mismatch, re-run both batches; skip if the plain baseline is
    unstable, fail only on a reproducible difference.
    """
    import pytest

    prompts = [TEXT] * 8

    def steered():
        steering = [
            steering_spec(scale=0.0, layers=LAYERS, exclude_prompt_positions=[i])
            for i in range(7)
        ] + [None]
        return gen(llm, prompts, steering=steering)

    mixed = steered()
    plain = gen(llm, prompts)
    if mixed == plain:
        return
    plain2 = gen(llm, prompts)
    if plain2 != plain:
        pytest.skip("engine batch-nondeterministic here; byte oracle invalid")
    assert steered() == plain2, (
        "zero-scale distinct configs changed some request (reproducible)"
    )
