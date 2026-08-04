# SPDX-License-Identifier: Apache-2.0
"""Tier-1 full-graph steering (steer_graph_mode=full).

The steering kernel `hidden += mask * vectors[row_tok]` reads persistent
buffers and captures into full CUDA graphs; triggers/routing are
computed host-side each step. Covers: full cudagraphs kept (no
piecewise downgrade); steering fires; scale-0 steering byte-identical
to no steering; byte-identical replay of a repeated steered run;
per-request routing isolation in a mixed [steered, plain] batch;
independently built identical v2 specs replaying byte-identically
(replay determinism across spec objects); loreft rejected as not
graph-safe.

All steering uses v2 SteeringSpec; STEER_TEST_EAGER=1 runs the same
kernel path eagerly (skipping the cudagraph-mode check).
"""

import os

import pytest
from vllm import SamplingParams

from helpers import DENSE_MODEL, steering_spec

EAGER = os.environ.get("STEER_TEST_EAGER", "0") == "1"

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    # No explicit steer_graph_mode: compiled engines resolve the "auto"
    # default to full — this module validates that default end to end.
    enforce_eager=EAGER,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.25,
    max_model_len=2048,
)
if EAGER:
    # The auto default resolves to piecewise under eager; the debug
    # escape hatch still needs the full-graph kernel path.
    ENGINE_KWARGS["steer_graph_mode"] = "full"

TEXT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
LAYERS = list(range(10, 26))
# ignore_eos keeps batch geometries aligned for byte-exact comparisons.
SP = SamplingParams(temperature=0.0, max_tokens=96, ignore_eos=True)


def happy_spec():
    return steering_spec(scale=2.0, layers=LAYERS)


def gen(llm, prompts, **kwargs):
    outs = llm.generate(prompts, sampling_params=SP, use_tqdm=False, **kwargs)
    return [o.outputs[0].text for o in outs]


@pytest.fixture(scope="module")
def outs(llm):
    """Single-prompt and mixed-batch outputs on one engine."""
    spec = happy_spec()
    return {
        "plain": gen(llm, [TEXT])[0],
        "zero": gen(llm, [TEXT], steering=steering_spec(scale=0.0,
                                                        layers=LAYERS))[0],
        "happy": gen(llm, [TEXT], steering=spec)[0],
        "happy2": gen(llm, [TEXT], steering=spec)[0],
        # Mixed batch: one steered + one plain request in ONE batch
        # (None entries in the steering sequence leave prompts unsteered).
        "batch_mixed": gen(
            llm,
            [TEXT, TEXT],
            steering=[happy_spec(), None],
        ),
        "batch_plain": gen(llm, [TEXT, TEXT]),
    }


@pytest.mark.skipif(EAGER, reason="STEER_TEST_EAGER=1 runs the kernel eagerly")
def test_full_cudagraphs_kept(llm):
    """Compiled engines resolve the auto default to full-graph steering
    and must not downgrade to piecewise graphs."""
    cfg = llm.llm_engine.vllm_config
    assert cfg.steer_vector_config.graph_mode == "full", (
        f"auto default resolved to {cfg.steer_vector_config.graph_mode!r}"
    )
    comp = cfg.compilation_config
    assert comp.cudagraph_mode.has_full_cudagraphs(), (
        f"cudagraph_mode {comp.cudagraph_mode} has no full graphs"
    )


def test_steering_fires(outs):
    assert outs["happy"] != outs["plain"], (
        "steered output identical to unsteered"
    )


def test_zero_scale_identical_to_no_steering(outs):
    assert outs["zero"] == outs["plain"], (
        "scale-0 steering differs from no steering"
    )


def test_repeated_steered_run_deterministic(outs):
    """Graph replay of the same spec object is byte-stable."""
    assert outs["happy"] == outs["happy2"], (
        "repeated steered run not deterministic"
    )


def test_mixed_batch_routing_isolation(outs):
    """A plain request beside a steered one stays byte-identical."""
    assert outs["batch_mixed"][1] == outs["batch_plain"][1], (
        "plain request contaminated in mixed batch"
    )


def test_fresh_identical_spec_replays_identically(llm, outs):
    """An independently built identical v2 spec is deterministic too.

    Replaces the v1-vs-v2 equivalence check of the original script: two
    identical v2-spec runs must be byte-identical and differ from the
    unsteered output.
    """
    again = gen(llm, [TEXT], steering=happy_spec())[0]
    assert again != outs["plain"], "v2 spec did not steer under full graphs"
    assert again == outs["happy"], (
        "identical v2 specs produced different outputs"
    )


def test_loreft_rejected_as_not_graph_safe(llm, outs):
    """Full-graph mode admits only direct/no-normalize/single-vector.

    Runs last (and after `outs` is built): loreft is a data-only
    algorithm, so the spec carries a minimal inline payload; the
    graph-safe rejection surfaces through the engine step and
    exception-type wrapping varies, so any raise counts (matching the
    original script). The underlying error is the admission-side
    "graph-safe configs" ValueError.
    """
    import numpy as np

    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
    from vllm.steer_vectors.payloads import ReftIntervention

    payload = ReftIntervention(
        rotate_layer=np.zeros((1536, 2), dtype=np.float32),
        learned_source_weight=np.zeros((1536, 2), dtype=np.float32),
        learned_source_bias=np.zeros(2, dtype=np.float32),
    )
    spec = SteeringSpec(
        vectors=[
            VectorSpec(
                data=payload,
                algorithm="loreft",
                scale=1.0,
                layers=LAYERS,
                apply=ApplySpec(phases=["prompt", "generation"]),
            )
        ]
    )
    with pytest.raises(Exception):
        gen(llm, [TEXT], steering=spec)
