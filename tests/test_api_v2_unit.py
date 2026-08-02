#!/usr/bin/env python3
"""CPU-only units for the v2 steering API (STEERING_API_V2.md).

Covers spec validation (explicit-failure paths), translation to the
internal engine struct, fingerprint/prefix-cache participation of
apply_spec, and the v2 position collector's semantics (exclusions always
compose, exact half-open generation window, negative positions from the
prompt end).
"""

import sys

import torch
from pydantic import ValidationError

from vllm.steer_vectors.api import (
    ApplySpec,
    SteeringSpec,
    VectorSpec,
    to_engine_request,
)
from vllm.steer_vectors.algorithms.triggers import (
    TriggerController,
    collect_positions_apply_spec,
)
from vllm.steer_vectors.request import (
    SteerVectorRequest,
    is_prompt_length_sensitive,
)
from vllm.steer_vectors.worker_manager import config_fingerprint

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"OK: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        FAILURES.append(name)


def expect_raises(name, exc_type, fn, needle=""):
    try:
        fn()
    except exc_type as e:
        check(name, needle in str(e), f"message: {e}")
    else:
        check(name, False, "did not raise")


VEC = "/tmp/does-not-need-to-exist.gguf"


def make_spec(**apply_kwargs):
    return SteeringSpec(
        vectors=[
            VectorSpec(
                source=VEC,
                scale=0.5,
                layers=[10],
                apply=ApplySpec(**apply_kwargs),
            )
        ]
    )


# --- U1: ApplySpec validation ------------------------------------------
expect_raises(
    "U1a empty phases rejected",
    ValidationError,
    lambda: ApplySpec(phases=[]),
    needle="non-empty",
)
expect_raises(
    "U1b duplicate phases rejected",
    ValidationError,
    lambda: ApplySpec(phases=["prompt", "prompt"]),
    needle="duplicates",
)
expect_raises(
    "U1c unknown phase rejected",
    ValidationError,
    lambda: ApplySpec(phases=["prefill"]),
)
expect_raises(
    "U1d -1 sentinel token rejected",
    ValidationError,
    lambda: ApplySpec(phases=["prompt"], tokens=[-1]),
    needle="real token ids",
)
expect_raises(
    "U1e window without generation phase rejected",
    ValidationError,
    lambda: ApplySpec(phases=["prompt"], generation_window=(0, 3)),
    needle="generation",
)
expect_raises(
    "U1f inverted window rejected",
    ValidationError,
    lambda: ApplySpec(phases=["generation"], generation_window=(3, 3)),
    needle="half-open",
)
expect_raises(
    "U1g empty tokens list rejected",
    ValidationError,
    lambda: ApplySpec(phases=["prompt"], tokens=[]),
    needle="non-empty",
)
check(
    "U1h unbounded window accepted",
    ApplySpec(phases=["generation"], generation_window=(2, None)) is not None,
)

# --- U2: VectorSpec / SteeringSpec validation --------------------------
apply_all = ApplySpec(phases=["prompt", "generation"])
expect_raises(
    "U2a path|algo hack rejected",
    ValidationError,
    lambda: VectorSpec(source="v.gguf|linear", apply=apply_all),
    needle="plain path",
)
expect_raises(
    "U2b unknown params rejected",
    ValidationError,
    lambda: VectorSpec(source=VEC, params={"beta": 1}, apply=apply_all),
    needle="unknown params",
)
expect_raises(
    "U2c source required for non-moe",
    ValidationError,
    lambda: VectorSpec(algorithm="direct", apply=apply_all),
    needle="requires a source",
)
expect_raises(
    "U2d moe without expert_ids rejected",
    ValidationError,
    lambda: VectorSpec(algorithm="moe_router", layers=[3], apply=apply_all),
    needle="expert_ids",
)
expect_raises(
    "U2e moe without layers rejected",
    ValidationError,
    lambda: VectorSpec(
        algorithm="moe_router", params={"expert_ids": [1]}, apply=apply_all
    ),
    needle="layers",
)
expect_raises(
    "U2f empty vectors rejected",
    ValidationError,
    lambda: SteeringSpec(vectors=[]),
    needle="non-empty",
)
expect_raises(
    "U2g multi-vector moe rejected",
    ValidationError,
    lambda: SteeringSpec(
        vectors=[
            VectorSpec(source=VEC, apply=apply_all),
            VectorSpec(
                algorithm="moe_router",
                layers=[3],
                params={"expert_ids": [1]},
                apply=apply_all,
            ),
        ]
    ),
    needle="moe_router",
)

# --- U3: translation to the engine struct ------------------------------
spec = make_spec(phases=["prompt"], exclude_positions=[0], tokens=[5])
req = to_engine_request(spec)
check("U3a single-vector path", req.steer_vector_local_path == VEC)
check("U3b scale/layers carried", req.scale == 0.5 and req.target_layers == [10])
check(
    "U3c apply_spec wire form",
    req.apply_spec
    == {
        "phases": ["prompt"],
        "tokens": [5],
        "positions": None,
        "exclude_tokens": None,
        "exclude_positions": [0],
        "window": None,
    },
    f"got {req.apply_spec}",
)
check("U3d no legacy trigger fields", req.prefill_trigger_tokens is None)

moe_req = to_engine_request(
    SteeringSpec(
        vectors=[
            VectorSpec(
                algorithm="moe_router",
                layers=[3],
                params={"expert_ids": [1, 2], "mode": "soft", "lambda": 0.7},
                apply=apply_all,
            )
        ]
    )
)
check(
    "U3e moe params folded",
    moe_req.moe_expert_ids == [1, 2]
    and moe_req.moe_mode == "soft"
    and moe_req.moe_lambda == 0.7
    and moe_req.moe_topk == 8,
)

multi = to_engine_request(
    SteeringSpec(
        vectors=[
            VectorSpec(source=VEC, layers=[10], apply=apply_all),
            VectorSpec(
                source=VEC,
                layers=[12],
                apply=ApplySpec(phases=["generation"]),
            ),
        ],
        conflict="sequential",
    )
)
check(
    "U3f multi-vector translation",
    multi.is_multi_vector
    and len(multi.vector_configs) == 2
    and multi.conflict_resolution == "sequential"
    and multi.vector_configs[1].apply_spec["phases"] == ["generation"],
)

# --- U4: engine-struct validation of apply_spec ------------------------
expect_raises(
    "U4a apply_spec + legacy triggers rejected",
    ValueError,
    lambda: SteerVectorRequest(
        steer_vector_name="x",
        steer_vector_int_id=7,
        steer_vector_local_path=VEC,
        prefill_trigger_tokens=[-1],
        apply_spec={"phases": ["prompt"]},
    ),
    needle="apply_spec",
)
expect_raises(
    "U4b malformed apply_spec rejected",
    ValueError,
    lambda: SteerVectorRequest(
        steer_vector_name="x",
        steer_vector_int_id=7,
        steer_vector_local_path=VEC,
        apply_spec={"phases": ["prompt"], "bogus": 1},
    ),
    needle="unknown keys",
)
check(
    "U4c apply_spec alone satisfies the trigger requirement",
    SteerVectorRequest(
        steer_vector_name="x",
        steer_vector_int_id=7,
        steer_vector_local_path=VEC,
        apply_spec={"phases": ["prompt"]},
    )
    is not None,
)

# --- U5: prompt-length sensitivity + fingerprint -----------------------
check(
    "U5a plain spec not length-sensitive",
    not is_prompt_length_sensitive(to_engine_request(make_spec(phases=["prompt"]))),
)
check(
    "U5b negative position is length-sensitive",
    is_prompt_length_sensitive(
        to_engine_request(make_spec(phases=["prompt"], positions=[-1]))
    ),
)
check(
    "U5c generation window is length-sensitive",
    is_prompt_length_sensitive(
        to_engine_request(
            make_spec(phases=["generation"], generation_window=(0, 2))
        )
    ),
)

fp_a1 = config_fingerprint(to_engine_request(make_spec(phases=["prompt"])))
fp_a2 = config_fingerprint(to_engine_request(make_spec(phases=["prompt"])))
fp_b = config_fingerprint(
    to_engine_request(make_spec(phases=["prompt"], exclude_positions=[0]))
)
check("U5d fingerprint stable across constructions", fp_a1 == fp_a2)
check("U5e fingerprint differs on apply_spec change", fp_a1 != fp_b)

# --- U6: v2 collector semantics (CPU tensors) --------------------------
# One request, prompt length 4, decoding: batch = 4 prompt tokens in one
# step is simulated per-scenario below with explicit samples_info.


def run_collector(spec_kwargs, token_ids, num_computed, is_decode, num_output,
                  num_prompt):
    wire = ApplySpec(**spec_kwargs).to_wire()
    tokens = torch.tensor(token_ids)
    info = {
        "query_start_loc": torch.tensor([0, len(token_ids)]),
        "num_computed": torch.tensor([num_computed]),
        "is_decode_mask": torch.tensor([is_decode]),
        "num_output_tokens": torch.tensor([num_output]),
        "num_prompt_tokens": torch.tensor([num_prompt]),
    }
    out = collect_positions_apply_spec(tokens, info, wire)
    return [] if out is None else out.tolist()


# prefill step: tokens abs 0..3, prompt len 4
check(
    "U6a prompt phase covers all prompt tokens",
    run_collector({"phases": ["prompt"]}, [11, 12, 13, 14], 0, False, 0, 4)
    == [0, 1, 2, 3],
)
check(
    "U6b exclusions compose with phase-wide prompt (v1 bypass gone)",
    run_collector(
        {"phases": ["prompt"], "exclude_positions": [0], "exclude_tokens": [13]},
        [11, 12, 13, 14],
        0,
        False,
        0,
        4,
    )
    == [1, 3],
)
check(
    "U6c generation phase skips prefill step",
    run_collector({"phases": ["generation"]}, [11, 12, 13, 14], 0, False, 0, 4)
    == [],
)
# decode step processing generated token j: num_output = j + 1
check(
    "U6d window (0,2) hits decode steps 0 and 1 only",
    [
        run_collector(
            {"phases": ["generation"], "generation_window": (0, 2)},
            [99],
            4 + j,
            True,
            j + 1,
            4,
        )
        for j in range(4)
    ]
    == [[0], [0], [], []],
)
check(
    "U6e window (1,None) skips only decode step 0",
    [
        run_collector(
            {"phases": ["generation"], "generation_window": (1, None)},
            [99],
            4 + j,
            True,
            j + 1,
            4,
        )
        for j in range(3)
    ]
    == [[], [0], [0]],
)
check(
    "U6f negative position resolves against prompt length",
    run_collector({"phases": ["prompt"], "positions": [-1]},
                  [11, 12], 2, False, 0, 4)
    == [1],
)
check(
    "U6g token and position triggers union",
    run_collector(
        {"phases": ["prompt"], "tokens": [11], "positions": [3]},
        [11, 12, 13, 14],
        0,
        False,
        0,
        4,
    )
    == [0, 3],
)

# --- U7: TriggerController integration ---------------------------------
ctrl = TriggerController()
ctrl.configure_from_dict(
    {"apply_spec": ApplySpec(phases=["prompt", "generation"]).to_wire()}
)
check("U7a controller global fast path", ctrl.is_global_only_config())
ctrl2 = TriggerController()
ctrl2.configure_from_dict(
    {"apply_spec": ApplySpec(phases=["prompt"], exclude_tokens=[3]).to_wire()}
)
check(
    "U7b filtered spec not global, has triggers",
    not ctrl2.is_global_only_config()
    and ctrl2.has_any_triggers()
    and ctrl2.has_prefill_triggers(),
)

print(f"\nOVERALL: {'FAIL' if FAILURES else 'PASS'}")
sys.exit(1 if FAILURES else 0)
