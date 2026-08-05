# SPDX-License-Identifier: Apache-2.0
"""Compiled-mode (piecewise) MoE gate steering on OLMoE-1B-7B.

The vllm::steer_moe_gate op is registered as a piecewise splitting op;
under compiled execution the gate steering runs eagerly between
CUDA-graph segments. Capture streams are eager-only, so the mechanism
is verified through the steering trace instead.

Coverage:
  - the engine boots compiled (no enforce_eager) with a moe_router
    config and generates; steered output differs from unsteered
  - the steering trace records gate applies at every MoE layer, with
    full prompt coverage on the prefill step
  - an unsteered generate leaves no apply records (routing off)
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
ALL_LAYERS = tuple(range(NUM_LAYERS))

ENGINE_KWARGS = dict(
    model=MODEL,
    enable_steer_vector=True,
    steer_algorithms=["moe_router"],
    enforce_eager=False,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.4,
    max_model_len=4096,
)

SP = SamplingParams(temperature=0.0, max_tokens=24)


@pytest.fixture(scope="module")
def prompt_ids(llm):
    tok = llm.get_tokenizer()
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Count to fifteen."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok(rendered, add_special_tokens=False).input_ids


@pytest.fixture(scope="module")
def deact_spec(tmp_path_factory):
    """Deactivate experts 0..19 at every MoE layer via a JSON config."""
    path = tmp_path_factory.mktemp("moe_compiled") / "compiled_deact.json"
    path.write_text(
        json.dumps(
            {
                "layer_configs": {
                    str(layer): {
                        "mode": "deactivate",
                        "expert_ids": list(range(20)),
                    }
                    for layer in ALL_LAYERS
                }
            }
        )
    )
    return steering_spec(
        source=str(path), algorithm="moe_router", scale=1.0, layers=None
    )


def test_unsteered_run_leaves_no_applies(trace, prompt_ids):
    """Without steering the gate routing is off: zero apply records."""
    out, by_layer = trace.run(prompt_ids, SP, layers=ALL_LAYERS)
    assert out.outputs[0].text, "compiled engine produced no output"
    applied = sorted(lid for lid, pos in by_layer.items() if pos)
    assert not applied, f"applies without steering at layers {applied}"


def test_steered_output_and_trace_cover_all_layers(
    trace, prompt_ids, deact_spec
):
    """Steering changes the output and applies at every MoE layer with
    full prompt coverage on the prefill step."""
    baseline, _ = trace.run(prompt_ids, SP, layers=ALL_LAYERS)
    steered, by_layer = trace.run(
        prompt_ids, SP, layers=ALL_LAYERS, steering=deact_spec
    )
    base_text = baseline.outputs[0].text
    steer_text = steered.outputs[0].text
    assert base_text and steer_text, "compiled engine produced no output"
    assert steer_text != base_text, f"both={steer_text!r}"

    layers_hit = {lid for lid, pos in by_layer.items() if pos}
    assert layers_hit == set(ALL_LAYERS), f"layers={sorted(layers_hit)}"
    want = set(range(len(prompt_ids)))
    for lid in ALL_LAYERS:
        prefill = {p for p, is_prefill in by_layer[lid] if is_prefill}
        assert prefill == want, (
            f"L{lid}: prefill coverage {sorted(prefill)} != all "
            f"{len(prompt_ids)} prompt tokens"
        )
