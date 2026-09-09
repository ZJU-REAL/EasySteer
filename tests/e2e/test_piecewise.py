# SPDX-License-Identifier: Apache-2.0
"""Tier-2 piecewise CUDA-graph steering (vllm::steer_apply).

The engine boots WITHOUT enforce_eager so the model is torch.compiled
and piecewise cudagraphs are captured. Covers: config wiring (the
steer_apply splitting op and piecewise cudagraph mode), steering firing
under compiled execution and scale-0 steering as a no-op on this workload.

The checks compare behavior within one compiled engine.
"""

import os

import pytest
from vllm import SamplingParams

from helpers import DENSE_MODEL, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_algorithms=["direct"],
    # Tier-2 is the point: this direct-only workload would auto-resolve
    # to in_graph on a compiled engine, so pin the split tier.
    steer_graph_mode="split",
    enforce_eager=False,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
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
SP = SamplingParams(temperature=0.0, max_tokens=128)


@pytest.fixture(scope="module")
def outs(llm):
    """Unsteered / zero-scale / steered outputs on one engine."""

    def gen(spec=None):
        out = llm.generate(
            TEXT, sampling_params=SP, use_tqdm=False, steering=spec
        )
        return out[0].outputs[0].text

    return {
        "plain": gen(),
        "zero": gen(steering_spec(scale=0.0, layers=LAYERS)),
        "happy": gen(steering_spec(scale=2.0, layers=LAYERS)),
    }


def test_piecewise_config_wiring(llm):
    """steer_apply is a splitting op and piecewise cudagraphs are on."""
    comp = llm.llm_engine.vllm_config.compilation_config
    assert "vllm::steer_apply" in (comp.splitting_ops or []), (
        f"vllm::steer_apply missing from splitting_ops: {comp.splitting_ops}"
    )
    assert "vllm::steer_moe_gate" not in (comp.splitting_ops or [])
    assert comp.cudagraph_mode.has_piecewise_cudagraphs(), (
        f"cudagraph_mode is {comp.cudagraph_mode}, expected piecewise"
    )


def test_steering_fires_under_compiled_execution(outs):
    """Spike proof that the steering hooks trace into the graph."""
    assert outs["happy"] != outs["zero"], (
        "steered output identical to unsteered (steering NOT firing "
        "under compiled execution)"
    )


def test_zero_scale_identical_to_no_steering(outs):
    """The op is a clean no-op on unsteered tokens."""
    assert outs["zero"] == outs["plain"], (
        "scale-0 steering differs from no steering"
    )
