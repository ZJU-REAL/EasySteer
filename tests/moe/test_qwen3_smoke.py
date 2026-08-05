# SPDX-License-Identifier: Apache-2.0
"""Qwen3-MoE architecture smoke test (single GPU, eager).

Second structural-discovery case beyond OLMoE: Qwen3-MoE passes its gate
into FusedMoE (internal-router path — the gate module is invoked inside
the MoE runner, not in the block forward), so this validates that gate
hooks fire on that call path for both steering and capture.

Coverage:
  - router_logits capture: every MoE layer, one row per prompt token,
    n_experts wide
  - a deactivate config forces the experts to the bottom of every
    captured row at every layer, and the engine generates normally
"""

import json
import os

import numpy as np
import pytest
from vllm import SamplingParams
from vllm.capture import deserialize_captured

from helpers import QWEN3_MODEL, steering_spec

# Qwen3-30B-A3B (no suffix) is the exact model evaluated in the SteerMoE
# paper; the -2507 refresh is a later release. The base model is a hybrid
# thinking model, so prompts render with enable_thinking=False (as the
# official SteerMoE code does).
MODEL = QWEN3_MODEL
with open(os.path.join(MODEL, "config.json")) as f:
    _hf_cfg = json.load(f)
NUM_LAYERS = _hf_cfg["num_hidden_layers"]
N_EXPERTS = _hf_cfg["num_experts"]
MOE_LAYERS = [
    layer
    for layer in range(NUM_LAYERS)
    if layer not in (_hf_cfg.get("mlp_only_layers") or [])
]

ENGINE_KWARGS = dict(
    model=MODEL,
    enable_steer_vector=True,
    steer_algorithms=["moe_router"],
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=float(os.environ.get("STEER_TEST_GPU_MEM", "0.92")),
    max_model_len=2048,
    max_num_seqs=4,
)

DEACT = list(range(10))


def rpc(llm, method, *args, **kwargs):
    return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)[0]


def captured(llm, prompt_ids, spec=None, max_tokens=1):
    """Post-steering router logits per layer plus the generated text."""
    rpc(llm, "start_capture", "router_logits")
    try:
        outs = llm.generate(
            {"prompt_token_ids": prompt_ids},
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=max_tokens
            ),
            steering=spec,
            use_tqdm=False,
        )
        raw = rpc(llm, "fetch_captured", "router_logits")
    finally:
        rpc(llm, "stop_capture", "router_logits")
    out = deserialize_captured(raw)[0]
    return (
        {lid: t.float().numpy() for lid, t in out.items()},
        outs[0].outputs[0].text,
    )


@pytest.fixture(scope="module")
def prompt_ids(llm):
    tok = llm.get_tokenizer()
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Count to ten."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tok(rendered, add_special_tokens=False).input_ids


def test_capture_on_internal_router_gate_path(llm, prompt_ids):
    """Every MoE layer captured, one row per prompt token, n_experts wide."""
    logits, _ = captured(llm, prompt_ids)
    assert sorted(logits) == MOE_LAYERS, (
        f"got {len(logits)} layers, expected {len(MOE_LAYERS)}"
    )
    shapes = {t.shape for t in logits.values()}
    want = (len(prompt_ids), N_EXPERTS)
    assert shapes == {want}, f"shapes={shapes} expected {want}"


def test_deactivate_on_internal_router_gate_path(llm, prompt_ids, tmp_path):
    """Deactivated experts sit at the bottom of every captured row and the
    engine generates normally under steering."""
    path = tmp_path / "qwen3_deact.json"
    path.write_text(
        json.dumps(
            {
                "layer_configs": {
                    str(layer): {"mode": "deactivate", "expert_ids": DEACT}
                    for layer in MOE_LAYERS
                }
            }
        )
    )
    spec = steering_spec(
        source=str(path), algorithm="moe_router", scale=1.0, layers=None
    )
    logits, text = captured(llm, prompt_ids, spec, max_tokens=8)
    assert logits, "no router logits captured under steering"
    others = np.setdiff1d(np.arange(N_EXPERTS), DEACT)
    for lid, rows in sorted(logits.items()):
        dominated = rows[:, DEACT].max(axis=-1) <= rows[:, others].min(axis=-1)
        assert dominated.all(), (
            f"L{lid}: rows {np.flatnonzero(~dominated).tolist()} not steered"
        )
    assert text, "engine produced no output under steering"
