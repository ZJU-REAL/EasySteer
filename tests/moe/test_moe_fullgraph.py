# SPDX-License-Identifier: Apache-2.0
"""MoE gate steering under full CUDA graphs (in_graph pinned;
moe_router is conditionally graph-safe, so auto resolves it to split).

The gate hook's captured kernel mirrors _transform_toggle (log-softmax,
activated experts to per-token max+eps, deactivated to min-eps) over
persistent expert toggle tables, so MoE models keep full CUDA graphs
while steering. Covers: full cudagraphs kept under the pinned
in-graph tier; a deactivation config changes the output; replay is
deterministic; an unsteered co-batched request is untouched; soft-mode
and file-based configs reject with the split-tier remedy.
"""

import json
import os

import pytest
from vllm import SamplingParams

from helpers import MOE_MODEL, steering_spec

MODEL = MOE_MODEL
with open(os.path.join(MODEL, "config.json")) as f:
    _hf_cfg = json.load(f)
NUM_LAYERS = _hf_cfg["num_hidden_layers"]

ENGINE_KWARGS = dict(
    model=MODEL,
    enable_steer_vector=True,
    steer_algorithms=["moe_router"],
    # moe_router is conditionally graph-safe (inline configs only), so
    # auto resolves it to split; this module tests the in-graph gate
    # kernel — pin the tier explicitly.
    steer_graph_mode="in_graph",
    enforce_eager=False,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.4,
    max_model_len=4096,
)

PROMPT = "The capital of France is"
SP = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)
DEACT = list(range(20))


def deact_spec():
    return steering_spec(
        source=None,
        algorithm="moe_router",
        scale=1.0,
        layers=list(range(NUM_LAYERS)),
        params={"expert_ids": DEACT, "mode": "deactivate"},
    )


def gen(llm, prompts, **kwargs):
    outs = llm.generate(prompts, sampling_params=SP, use_tqdm=False, **kwargs)
    return [o.outputs[0].text for o in outs]


def test_full_cudagraphs_kept(llm):
    cfg = llm.llm_engine.vllm_config
    assert cfg.steer_vector_config.graph_mode == "in_graph"
    assert cfg.compilation_config.cudagraph_mode.has_full_cudagraphs()


def test_deactivation_steers_and_replays(llm):
    plain = gen(llm, [PROMPT])[0]
    steered = gen(llm, [PROMPT], steering=deact_spec())[0]
    steered2 = gen(llm, [PROMPT], steering=deact_spec())[0]
    assert steered != plain, "expert deactivation did not change the output"
    assert steered == steered2, "steered replay not deterministic"


def test_mixed_batch_isolation(llm):
    batch_plain = gen(llm, [PROMPT, PROMPT])
    mixed = gen(llm, [PROMPT, PROMPT], steering=[deact_spec(), None])
    assert mixed[1] == batch_plain[1], (
        "plain request contaminated by co-batched moe steering"
    )


def test_soft_mode_and_file_config_rejected(llm):
    with pytest.raises(Exception):
        gen(llm, [PROMPT], steering=steering_spec(
            source=None, algorithm="moe_router", scale=1.0, layers=[0],
            params={"expert_ids": [1], "mode": "soft"},
        ))
    with pytest.raises(Exception):
        gen(llm, [PROMPT], steering=steering_spec(
            source=os.path.join(MODEL, "config.json"),
            algorithm="moe_router", scale=1.0, layers=None,
        ))
