# SPDX-License-Identifier: Apache-2.0
"""Tier-2 piecewise CUDA-graph steering (vllm::steer_apply).

The engine boots WITHOUT enforce_eager so the model is torch.compiled
and piecewise cudagraphs are captured. Covers: config wiring (the
steer_apply splitting op and piecewise cudagraph mode), steering firing
under compiled execution, scale-0 steering being byte-identical to no
steering, and byte-identical replay of a repeated steered run.

Compiled kernels drift numerically from eager, so there is no
byte-compare against an eager golden (the original script printed that
diff as informational only); the checks here are behavior-level within
one compiled engine.
"""

import os

import pytest
from vllm import SamplingParams

from helpers import DENSE_MODEL, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_graph_mode="piecewise",  # Tier-2 is the point (default is full)
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
    """Unsteered / zero-scale / twice-steered outputs on one engine."""

    def gen(spec=None):
        out = llm.generate(
            TEXT, sampling_params=SP, use_tqdm=False, steering=spec
        )
        return out[0].outputs[0].text

    return {
        "plain": gen(),
        "zero": gen(steering_spec(scale=0.0, layers=LAYERS)),
        "happy": gen(steering_spec(scale=2.0, layers=LAYERS)),
        "happy2": gen(steering_spec(scale=2.0, layers=LAYERS)),
    }


def test_piecewise_config_wiring(llm):
    """steer_apply is a splitting op and piecewise cudagraphs are on."""
    comp = llm.llm_engine.vllm_config.compilation_config
    assert "vllm::steer_apply" in (comp.splitting_ops or []), (
        f"vllm::steer_apply missing from splitting_ops: {comp.splitting_ops}"
    )
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


def test_repeated_steered_run_deterministic(outs):
    """Cudagraph replay of an identical steered config is byte-stable."""
    assert outs["happy"] == outs["happy2"], (
        "repeated steered run not deterministic"
    )
