# SPDX-License-Identifier: Apache-2.0
"""MoE gate steering under full CUDA graphs (in_graph pinned;
moe_router is conditionally graph-safe, so auto resolves it to split).

The gate hook's captured kernel mirrors _transform_toggle (log-softmax,
activated experts to per-token max+eps, deactivated to min-eps) over
persistent expert toggle tables, so MoE models keep full CUDA graphs
while steering. Covers: full cudagraphs kept under the pinned
in-graph tier; a deactivation config changes the output; mixed traffic
completes with a steering effect; soft, soft_topk and file-based configs
are accepted. Random sampling remains split-only. Fixed-input kernel tests
check the exact transforms while this suite checks real model integration.
"""

import json
import os

import pytest
import torch
from helpers import MOE_MODEL, steering_spec
from vllm import SamplingParams

MODEL = MOE_MODEL
with open(os.path.join(MODEL, "config.json")) as f:
    _hf_cfg = json.load(f)
NUM_LAYERS = _hf_cfg["num_hidden_layers"]

ENGINE_KWARGS = dict(
    model=MODEL,
    enable_steer_vector=True,
    steer_algorithms=["moe_router"],
    # moe_router is conditionally graph-safe (random mode needs split), so
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


def test_deactivation_steers(llm):
    plain = gen(llm, [PROMPT])[0]
    steered = gen(llm, [PROMPT], steering=deact_spec())[0]
    assert steered != plain, "expert deactivation did not change the output"


def test_mixed_batch_completes_and_steers(llm):
    mixed = gen(llm, [PROMPT, PROMPT], steering=[deact_spec(), None])
    assert len(mixed) == 2 and all(mixed)
    assert mixed[0] != mixed[1], "the mixed-batch steering effect is missing"


@pytest.mark.parametrize("mode", ["soft", "soft_topk"])
def test_soft_modes_run_in_graph(llm, mode):
    spec = steering_spec(
        source=None, algorithm="moe_router", scale=1.0, layers=[0],
        params={"expert_ids": [1, 3], "mode": mode, "lambda": 1.5, "topk": 3},
    )
    assert gen(llm, [PROMPT], steering=spec)[0]


def test_file_config_runs_in_graph(llm, tmp_path):
    path = tmp_path / "deactivate.json"
    path.write_text(json.dumps({
        "layer_configs": {
            str(layer): {"expert_ids": DEACT, "mode": "deactivate"}
            for layer in range(NUM_LAYERS)
        }
    }))
    spec = steering_spec(
        source=str(path), algorithm="moe_router", scale=1.0, layers=None,
    )
    mixed = gen(llm, [PROMPT, PROMPT], steering=[spec, None])
    assert all(mixed) and mixed[0] != mixed[1]


def test_router_capture_replays_full_graph_with_steering(llm):
    """Default, disabled and overridden router payloads share capture graphs."""
    from vllm.capture import deserialize_captured, match_capture_request_id
    from vllm.steer_vectors import (
        ApplySpec, RouterConfig, SelectSpec, SteeringSpec, VectorSpec,
    )

    def router_spec(mode, expert_ids):
        return SteeringSpec(vectors=[VectorSpec(
            data=RouterConfig({0: {"mode": mode, "expert_ids": expert_ids}}),
            algorithm="moe_router", apply=ApplySpec(generation="all"),
        )])

    def rpc(method, *args, **kwargs):
        return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)[0]

    stream = "router_logits"
    before = rpc("capture_status", stream)
    llm.set_default_steering(router_spec("deactivate", DEACT))
    rpc(
        "start_capture", stream, layers=[0],
        select=SelectSpec(generation="all").to_wire(),
    )
    try:
        outputs = llm.generate(
            [PROMPT] * 3,
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=4, ignore_eos=True,
            ),
            steering=[None, False, router_spec("activate", [31])],
            use_tqdm=False,
        )
        after = rpc("capture_status", stream)
        raw = rpc("fetch_captured", stream, clear=True)
    finally:
        rpc("stop_capture", stream)
        llm.set_default_steering(None)

    assert after["graph_ready"] and after["graph_buffer_bytes"] > 0
    assert after["graph_replays"] >= before["graph_replays"] + 3
    assert after["meta_complete"] and after["tokens_dropped"] == 0
    tensors, meta = deserialize_captured(raw)
    assert set(tensors) == {0} and set(meta) == {0}
    logits, labels = tensors[0], meta[0]
    assert tuple(logits.shape) == (9, _hf_cfg["num_experts"])
    assert torch.isfinite(logits).all().item()
    by_request = []
    for output in outputs:
        indices = [
            i for i, rid in enumerate(labels.req_ids)
            if match_capture_request_id(rid, output.request_id)
        ]
        generated = list(output.outputs[0].token_ids)
        prompt_len = len(output.prompt_token_ids)
        assert len(generated) == 4 and len(indices) == 3
        assert labels.positions[indices].tolist() == list(
            range(prompt_len, prompt_len + 3)
        )
        assert labels.token_ids[indices].tolist() == generated[:3]
        by_request.append(logits[indices])
    inherited, disabled, overridden = by_request
    deactivated = inherited[:, DEACT]
    assert torch.equal(deactivated, deactivated[:, :1].expand_as(deactivated))
    # BF16 can round away epsilon, so equality with the other minimum is valid.
    assert (deactivated[:, 0] <= inherited[:, len(DEACT):].min(dim=-1).values).all()
    assert not torch.equal(
        disabled[:, DEACT], disabled[:, :1].expand(-1, len(DEACT)),
    ), "False must bypass the default expert deactivation"
    assert (overridden[:, 31] >= overridden.max(dim=-1).values).all()
