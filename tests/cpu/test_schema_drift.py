# SPDX-License-Identifier: Apache-2.0
"""Drift guard between the user schema and the engine wire structs.

The pydantic schema (api.SteeringSpec/VectorSpec/ApplySpec) and the
msgspec IPC structs (request.SteeringRequest/ResolvedVector) are two
deliberate layers with one translation point, to_engine_request. This
suite makes silent drift impossible: every user-facing field must be
enumerated in the mapping below AND observably reach the wire struct.
Adding a schema field without wiring it (and updating the map) fails
here first.
"""

import msgspec
import numpy as np
from vllm.model_hooks.steering.api import to_engine_request
from vllm.model_hooks.steering.payloads import DirectionVector, from_wire
from vllm.model_hooks.steering.request import SteeringRequest
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

# User field -> how it is observed on the wire (documentation + guard).
VECTOR_SPEC_FIELDS = {
    "source": "ResolvedVector.source",
    "data": "ResolvedVector.payload (including content sha256)",
    "algorithm": "algorithm",
    "scale": "scale",
    "layers": "target_layers",
    "normalize": "normalize",
    "apply": "apply_spec",
    "params": "canonical router layer configuration in payload",
    "name": "steer_vector_name (single-vector label only)",
}
STEERING_SPEC_FIELDS = {
    "vectors": "SteeringRequest.vectors",
    "conflict": "conflict_resolution",
}


def engine_wire(spec):
    """Exercise the typed IPC round trip after the public-to-internal conversion."""
    return msgspec.msgpack.decode(
        msgspec.msgpack.encode(to_engine_request(spec)), type=SteeringRequest
    )


def test_every_user_field_is_enumerated():
    assert set(VectorSpec.model_fields) == set(VECTOR_SPEC_FIELDS), (
        "VectorSpec fields changed; wire the new field through "
        "to_engine_request and update VECTOR_SPEC_FIELDS"
    )
    assert set(SteeringSpec.model_fields) == set(STEERING_SPEC_FIELDS), (
        "SteeringSpec fields changed; wire the new field through "
        "to_engine_request and update STEERING_SPEC_FIELDS"
    )


def test_single_vector_fields_reach_the_wire():
    payload = DirectionVector({7: np.ones(8, dtype=np.float32)})
    spec = SteeringSpec(
        vectors=[VectorSpec(
            data=payload,
            algorithm="direct",
            scale=1.5,
            layers=[7],
            normalize=True,
            name="drift-check",
            apply=ApplySpec(
                prompt_positions=[-1], prompt_tokens=[5], exclude_prompt_positions=[0]
            ),
        )],
    )
    req = engine_wire(spec)
    assert req.vectors[0].algorithm == "direct"
    assert req.vectors[0].scale == 1.5
    assert req.vectors[0].target_layers == [7]
    assert req.vectors[0].normalize is True
    assert req.steer_vector_name == "drift-check"
    assert req.vectors[0].source == ""
    assert req.vectors[0].payload == payload.to_wire()
    assert req.vectors[0].payload["kind"] == "direction"
    assert req.vectors[0].payload["sha256"] == req.vectors[0].payload_sha256
    assert req.vectors[0].apply_spec == {
        "prompt": None,
        "generation": None,
        "prompt_tokens": [5],
        "prompt_positions": [-1],
        "prompt_window": None,
        "generation_tokens": None,
        "generation_positions": None,
        "generation_window": None,
        "exclude_prompt_tokens": None,
        "exclude_prompt_positions": [0],
        "exclude_prompt_window": None,
        "exclude_generation_tokens": None,
        "exclude_generation_positions": None,
        "exclude_generation_window": None,
    }


def test_multi_vector_fields_reach_the_wire():
    first = DirectionVector({3: np.ones(8, dtype=np.float32)})
    second = DirectionVector({4: np.zeros(8, dtype=np.float32)})
    spec = SteeringSpec(
        conflict="sequential",
        vectors=[
            VectorSpec(data=first, scale=0.5, layers=[3],
                       apply=ApplySpec(generation="all")),
            VectorSpec(data=second, scale=2.0, layers=[4],
                       apply=ApplySpec(prompt="all", generation="all")),
        ],
    )
    req = engine_wire(spec)
    assert len(req.vectors) == 2
    assert req.conflict_resolution == "sequential"
    assert [vc.scale for vc in req.vectors] == [0.5, 2.0]
    assert [vc.target_layers for vc in req.vectors] == [[3], [4]]
    assert [vc.payload for vc in req.vectors] == [first.to_wire(), second.to_wire()]
    assert len({vc.payload_sha256 for vc in req.vectors}) == 2
    assert req.vectors[0].apply_spec["generation"] == "all"
    assert req.vectors[0].apply_spec["prompt"] is None


def test_moe_params_reach_the_wire():
    for params in (
        {"expert_ids": [1, 5], "mode": "deactivate"},
        {"expert_ids": [1, 2], "mode": "soft", "lambda": 0.7},
    ):
        spec = SteeringSpec(vectors=[VectorSpec(
            algorithm="moe_router", scale=1.0, layers=[2], params=params,
            apply=ApplySpec(prompt="all", generation="all"),
        )])
        req = engine_wire(spec)
        assert req.vectors[0].payload["kind"] == "router"
        layer = from_wire(req.vectors[0].payload).layers[2]
        assert all(layer[key] == value for key, value in params.items())
