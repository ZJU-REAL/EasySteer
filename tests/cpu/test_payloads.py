# SPDX-License-Identifier: Apache-2.0
"""CPU units for in-memory steering payloads and the data= admission path."""

import numpy as np
import pytest
import torch
from vllm.model_hooks.steering.api import (
    ApplySpec,
    SteeringSpec,
    VectorSpec,
    to_engine_request,
)
from vllm.model_hooks.steering.payloads import (
    ConceptPair,
    DirectionVector,
    LinearMap,
    LowRankProjector,
    ReftIntervention,
    RouterConfig,
    from_wire,
    materialize,
)

APPLY = ApplySpec(prompt="all", generation="all")


def spec_of(vector: VectorSpec) -> SteeringSpec:
    return SteeringSpec(vectors=[vector])


class TestPayloadStructures:
    def test_direction_roundtrip_and_hash_determinism(self):
        dv = DirectionVector({10: np.ones(8), 11: torch.arange(12.0)})
        wire = dv.to_wire()
        again = DirectionVector({10: np.ones(8), 11: np.arange(12.0)}).to_wire()
        assert wire["sha256"] == again["sha256"]
        out = materialize(wire, "cpu", torch.float16, None)
        assert set(out) == {10, 11}
        assert out[10].shape == (8,) and out[11].shape == (12,)
        assert out[10].dtype == torch.float16
        assert out[11][3] == 3

    def test_hash_changes_with_content(self):
        a = DirectionVector({0: np.ones(4)}).to_wire()
        b = DirectionVector({0: np.ones(4) * 2}).to_wire()
        c = DirectionVector({1: np.ones(4)}).to_wire()
        assert len({a["sha256"], b["sha256"], c["sha256"]}) == 3

    def test_broadcast_kinds_require_layers(self):
        lm = LowRankProjector(np.ones((8, 2)), np.ones((8, 2)))
        with pytest.raises(ValueError, match="layers is required"):
            materialize(lm.to_wire(), "cpu", torch.float32, None)
        out = materialize(lm.to_wire(), "cpu", torch.float32, [3, 5])
        assert set(out) == {3, 5}
        assert out[3]["projector1"].shape == (8, 2)

    def test_linear_materializes_weight_and_bias(self):
        wire = LinearMap(np.eye(4), np.ones(4)).to_wire()
        out = materialize(wire, "cpu", torch.float32, [2])
        assert torch.equal(out[2]["weight"], torch.eye(4))
        assert torch.equal(out[2]["bias"], torch.ones(4))

    def test_concept_pair_role_names_enforced(self):
        with pytest.raises(ValueError, match="layers"):
            ConceptPair({1: np.ones(4)}, {2: np.ones(4)})
        wire = ConceptPair({4: np.ones(4)}, {4: np.zeros(4)}).to_wire()
        out = materialize(wire, "cpu", torch.float32, None)
        assert out[4]["h1"].sum() == 4 and out[4]["h2"].sum() == 0

    def test_reft_checkpoint_layer_wins(self):
        rf = ReftIntervention(np.ones((8, 2)), np.ones((2, 8)), layer=7)
        assert set(materialize(rf.to_wire(), "cpu", torch.float32, None)) == {7}

    @pytest.mark.parametrize("payload,algorithm,component", [
        (DirectionVector({2: np.ones(8), 5: np.ones(8)}), "direct", "hidden_states"),
        (ConceptPair({2: np.ones(8)}, {2: np.zeros(8)}),
         "concept_replace", "hidden_states"),
        (ReftIntervention(np.ones((8, 2)), np.ones((2, 8)), layer=2),
         "loreft", "hidden_states"),
        (RouterConfig({2: {"expert_ids": [0]}, 5: {"expert_ids": [1]}}),
         "moe_router", "router_logits"),
    ])
    def test_layer_filter_matches_admission_and_does_not_retarget_checkpoint(
        self, payload, algorithm, component
    ):
        from vllm.model_hooks.steering.validation import validate_request_model

        for targets, expected in (([2, 7], {2}), ([7], set())):
            request = to_engine_request(spec_of(VectorSpec(
                data=payload, algorithm=algorithm, layers=targets, apply=APPLY,
            )))
            installed = materialize(payload.to_wire(), "cpu", torch.float32, targets)
            assert set(installed) == expected
            if expected:
                validate_request_model(request, 8, {component: {2: 8}})
            else:
                with pytest.raises(ValueError, match="targets no modules"):
                    validate_request_model(request, 8, {component: {2: 8}})

    def test_validation_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="non-finite"):
            DirectionVector({0: np.array([1.0, np.inf])})
        with pytest.raises(ValueError, match="bias size"):
            LinearMap(np.ones((4, 4)), np.ones(5))
        with pytest.raises(ValueError, match="1-D"):
            DirectionVector({0: np.ones((2, 2))})
        with pytest.raises(ValueError, match="square"):
            LinearMap(np.ones((4, 1)))
        with pytest.raises(ValueError, match="hidden dimensions disagree"):
            ConceptPair({0: np.ones(4)}, {0: np.ones(1)})
        with pytest.raises(ValueError, match="2-D"):
            LowRankProjector(np.ones((1, 4, 2)), np.ones((1, 4, 2)))

    @pytest.mark.parametrize(
        "weight_shape,bias_size,pattern",
        [
            ((2, 7), 2, "learned_source_weight shape"),
            ((3, 8), 3, "learned_source_weight shape"),
            ((2, 8), 1, "rotation rank 2"),
        ],
    )
    def test_reft_rejects_incompatible_basis_source_and_bias(
        self, weight_shape, bias_size, pattern
    ):
        with pytest.raises(ValueError, match=pattern):
            ReftIntervention(
                np.ones((8, 2)), np.ones(weight_shape), np.ones(bias_size)
            )

    @pytest.mark.parametrize(
        "payload,algorithm,width",
        [
            (DirectionVector({0: np.ones(8)}), "direct", 8),
            (LinearMap(np.eye(8), np.ones(8)), "linear", 8),
            (LowRankProjector(np.ones((8, 2)), np.ones((8, 2))), "lm_steer", 8),
            (ReftIntervention(np.ones((8, 2)), np.ones((2, 8)), np.ones(2)), "loreft", 8),
            (ConceptPair({0: np.ones(8)}, {0: np.zeros(8)}), "concept_replace", 8),
            (DirectionVector({0: np.ones(12)}), "attention_add", 12),
        ],
        ids=["direction", "linear", "lowrank", "reft", "concept_pair", "attention"],
    )
    def test_model_width_checked_before_payload_device_allocation(
        self, payload, algorithm, width
    ):
        from types import SimpleNamespace
        from unittest.mock import patch

        from vllm.model_hooks.steering.capabilities import algorithm_target
        from vllm.model_hooks.steering.payload_cache import PayloadCache
        from vllm.model_hooks.steering.validation import validate_request_model

        config = SimpleNamespace(max_steer_vectors=1, adapter_dtype=torch.float32)
        wire = payload.to_wire()
        request = to_engine_request(spec_of(VectorSpec(
            data=payload, algorithm=algorithm, layers=[0], apply=APPLY,
        )))
        component = algorithm_target(algorithm)
        for hidden_size in (4, 16):
            cache = PayloadCache("cpu", config)
            with patch(
                "vllm.model_hooks.steering.payload_cache.materialize",
                side_effect=AssertionError("allocated invalid payload"),
            ), pytest.raises(ValueError, match=f"(?:hidden size|component width) {hidden_size}"):
                validate_request_model(request, hidden_size, {component: {0: hidden_size}})
                cache.get(wire, target_layers=[0])
            assert not cache._entries
        validate_request_model(request, 8, {component: {0: width}})
        cache = PayloadCache("cpu", config)
        assert set(cache.get(wire, target_layers=[0])) == {0}

    def test_wire_rejects_forged_identity_and_json_roundtrips(self):
        import copy

        from easysteer.vectors import to_json_payload

        payload = DirectionVector({0: np.arange(4.0)})
        wire = payload.to_wire()
        assert from_wire(to_json_payload(payload)).to_wire() == wire
        changed = copy.deepcopy(wire)
        changed["tensors"]["layer.0"]["data"] = np.ones(4, dtype=np.float32).tobytes()
        with pytest.raises(ValueError, match="sha256 does not match"):
            from_wire(changed)

        malformed = payload.to_wire()
        malformed["tensors"]["layer.0"]["shape"] = [-1]
        with pytest.raises(ValueError, match="invalid shape"):
            from_wire(malformed)

    def test_router_wire_is_canonical_and_detached_from_mutable_input(self):
        first = RouterConfig({5: {"expert_ids": [2], "mode": "soft_topk"}})
        reordered = RouterConfig({"5": {"mode": "soft_topk", "expert_ids": [2]}})
        wire = first.to_wire()
        assert wire == reordered.to_wire()
        restored = from_wire(wire)
        first.layers[5]["expert_ids"].append(7)
        wire["extra"]["layers"]["5"]["expert_ids"].append(9)
        assert restored.layers[5]["expert_ids"] == [2]

    @pytest.mark.parametrize(
        "params",
        [
            {"expert_ids": [-1]},
            {"expert_ids": [True]},
            {"expert_ids": [1], "epsilon": float("nan")},
            {"mode": "soft", "expert_ids": [1], "lambda": float("inf")},
            {"mode": "soft_topk", "expert_ids": [1], "topk": 0},
        ],
    )
    def test_router_rejects_invalid_parameters_before_materialization(self, params):
        with pytest.raises(ValueError):
            RouterConfig({0: params})


class TestDataAdmission:
    def test_data_and_source_mutually_exclusive(self):
        dv = DirectionVector({10: np.ones(8)})
        with pytest.raises(Exception, match="mutually exclusive"):
            VectorSpec(data=dv, source="x.gguf", apply=APPLY)

    def test_kind_must_match_algorithm(self):
        dv = DirectionVector({10: np.ones(8)})
        with pytest.raises(Exception, match="requires a 'lowrank'"):
            VectorSpec(data=dv, algorithm="lm_steer", layers=[1], apply=APPLY)

    def test_broadcast_payload_requires_layers_at_spec(self):
        lm = LowRankProjector(np.ones((8, 2)), np.ones((8, 2)))
        with pytest.raises(Exception, match="layers is required"):
            VectorSpec(data=lm, algorithm="lm_steer", apply=APPLY)

    def test_fingerprints_differ_by_payload_content(self):
        from vllm.model_hooks.steering.request import config_fingerprint

        def req_for(vec):
            return to_engine_request(
                spec_of(VectorSpec(data=vec, algorithm="direct", apply=APPLY))
            )

        fp1 = config_fingerprint(req_for(DirectionVector({0: np.ones(4)})))
        fp2 = config_fingerprint(req_for(DirectionVector({0: np.ones(4) * 2})))
        fp3 = config_fingerprint(req_for(DirectionVector({0: np.ones(4)})))
        assert fp1 != fp2
        assert fp1 == fp3

class TestEngineHeuristicsGone:
    def test_direct_rejects_pt_files(self):
        from vllm.model_hooks.steering.loading import resolve_vector_payload

        with pytest.raises(ValueError, match="only loads .gguf"):
            resolve_vector_payload("v.pt", None, "direct")

    def test_dataonly_algorithms_reject_paths(self):
        from vllm.model_hooks.steering.loading import resolve_vector_payload

        with pytest.raises(ValueError, match="data="):
            resolve_vector_payload("gpt2.pt", None, "lm_steer")
