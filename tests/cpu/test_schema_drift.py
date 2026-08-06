# SPDX-License-Identifier: Apache-2.0
"""Drift guard between the user schema and the engine wire structs.

The v2 pydantic schema (api.SteeringSpec/VectorSpec/ApplySpec) and the
msgspec IPC structs (request.SteerVectorRequest/VectorConfig) are two
deliberate layers with one translation point, to_engine_request. This
suite makes silent drift impossible: every user-facing field must be
enumerated in the mapping below AND observably reach the wire struct.
Adding a schema field without wiring it (and updating the map) fails
here first.
"""

import numpy as np

from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
from vllm.steer_vectors.api import to_engine_request
from vllm.steer_vectors.payloads import DirectionVector

# User field -> how it is observed on the wire (documentation + guard).
VECTOR_SPEC_FIELDS = {
    "source": "steer_vector_local_path / VectorConfig.path",
    "data": "inline_payload + payload_sha256",
    "algorithm": "algorithm",
    "scale": "scale",
    "layers": "target_layers",
    "normalize": "normalize",
    "apply": "apply_spec",
    "params": "moe_* fields (moe_router) / algorithm params",
    "name": "steer_vector_name (single-vector label only)",
}
STEERING_SPEC_FIELDS = {
    "vectors": "single-vector fields or vector_configs",
    "conflict": "conflict_resolution",
}


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
            apply=ApplySpec(prompt_positions=[-1]),
        )],
    )
    req = to_engine_request(spec)
    assert req.algorithm == "direct"
    assert req.scale == 1.5
    assert req.target_layers == [7]
    assert req.normalize is True
    assert req.steer_vector_name == "drift-check"
    assert req.inline_payload["sha256"] == req.payload_sha256
    assert req.apply_spec["prompt"] is None
    assert req.apply_spec["generation"] is None
    assert req.apply_spec["prompt_positions"] == [-1]


def test_multi_vector_fields_reach_the_wire():
    payload = DirectionVector({3: np.ones(8, dtype=np.float32)})
    spec = SteeringSpec(
        conflict="sequential",
        vectors=[
            VectorSpec(data=payload, scale=0.5, layers=[3],
                       apply=ApplySpec(generation="all")),
            VectorSpec(data=payload, scale=2.0, layers=[4],
                       apply=ApplySpec(prompt="all", generation="all")),
        ],
    )
    req = to_engine_request(spec)
    assert req.is_multi_vector
    assert req.conflict_resolution == "sequential"
    assert [vc.scale for vc in req.vector_configs] == [0.5, 2.0]
    assert [vc.target_layers for vc in req.vector_configs] == [[3], [4]]
    assert req.vector_configs[0].apply_spec["generation"] == "all"
    assert req.vector_configs[0].apply_spec["prompt"] is None


def test_moe_params_reach_the_wire():
    spec = SteeringSpec(vectors=[VectorSpec(
        algorithm="moe_router", scale=1.0, layers=[2],
        params={"expert_ids": [1, 5], "mode": "deactivate"},
        apply=ApplySpec(prompt="all", generation="all"),
    )])
    req = to_engine_request(spec)
    assert req.moe_expert_ids == [1, 5]
    assert req.moe_mode == "deactivate"
