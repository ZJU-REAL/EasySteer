# SPDX-License-Identifier: Apache-2.0
"""CPU units for in-memory steering payloads and the data= admission path."""

import numpy as np
import pytest
import torch

from vllm.steer_vectors.api import ApplySpec, SteeringSpec, VectorSpec, to_engine_request
from vllm.steer_vectors.payloads import (
    ConceptPair,
    DirectionVector,
    LinearMap,
    LowRankProjector,
    ReftIntervention,
    materialize,
)

APPLY = ApplySpec(phases=["prompt", "generation"])


def spec_of(vector: VectorSpec) -> SteeringSpec:
    return SteeringSpec(vectors=[vector])


class TestPayloadStructures:
    def test_direction_roundtrip_and_hash_determinism(self):
        dv = DirectionVector({10: np.ones(8), 11: torch.arange(8.0)})
        wire = dv.to_wire()
        again = DirectionVector({10: np.ones(8), 11: np.arange(8.0)}).to_wire()
        assert wire["sha256"] == again["sha256"]
        out = materialize(wire, "cpu", torch.float16, None)
        assert set(out) == {10, 11}
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

    def test_validation_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="non-finite"):
            DirectionVector({0: np.array([1.0, np.inf])})
        with pytest.raises(ValueError, match="bias size"):
            LinearMap(np.ones((4, 4)), np.ones(5))
        with pytest.raises(ValueError, match="1-D"):
            DirectionVector({0: np.ones((2, 2))})


class TestDataAdmission:
    def test_data_spec_builds_engine_request(self):
        dv = DirectionVector({10: np.ones(8)})
        req = to_engine_request(
            spec_of(VectorSpec(data=dv, algorithm="direct", apply=APPLY))
        )
        assert req.steer_vector_local_path == ""
        assert req.inline_payload["sha256"] == req.payload_sha256
        assert req.inline_payload["kind"] == "direction"

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

    def test_multi_vector_carries_payloads(self):
        dv = DirectionVector({10: np.ones(8)})
        spec = SteeringSpec(
            vectors=[
                VectorSpec(data=dv, algorithm="direct", apply=APPLY),
                VectorSpec(
                    data=DirectionVector({11: np.ones(8)}),
                    algorithm="direct",
                    apply=APPLY,
                ),
            ]
        )
        req = to_engine_request(spec)
        shas = {vc.payload_sha256 for vc in req.vector_configs}
        assert len(shas) == 2 and None not in shas

    def test_fingerprints_differ_by_payload_content(self):
        from vllm.steer_vectors.worker_manager import config_fingerprint

        def req_for(vec):
            return to_engine_request(
                spec_of(VectorSpec(data=vec, algorithm="direct", apply=APPLY))
            )

        fp1 = config_fingerprint(req_for(DirectionVector({0: np.ones(4)})))
        fp2 = config_fingerprint(req_for(DirectionVector({0: np.ones(4) * 2})))
        fp3 = config_fingerprint(req_for(DirectionVector({0: np.ones(4)})))
        assert fp1 != fp2
        assert fp1 == fp3

    def test_source_still_required_without_data(self):
        with pytest.raises(Exception, match="source file or an in-memory"):
            VectorSpec(algorithm="direct", apply=APPLY)


class TestEngineHeuristicsGone:
    def test_direct_rejects_pt_files(self):
        from vllm.steer_vectors.algorithms.direct import DirectAlgorithm

        with pytest.raises(ValueError, match="only loads .gguf"):
            DirectAlgorithm.load_from_path("v.pt", "cpu", config=None)

    def test_dataonly_algorithms_reject_paths(self):
        from vllm.steer_vectors.algorithms.lm_steer import LMSteerAlgorithm

        with pytest.raises(ValueError, match="data="):
            LMSteerAlgorithm.load_from_path("gpt2.pt", "cpu", config=None)

    def test_moe_mode_validator_is_shared(self):
        from vllm.steer_vectors.algorithms.moe_router import MoERouterAlgorithm

        assert MoERouterAlgorithm.validate_mode("boost") == "activate"
        with pytest.raises(ValueError, match="unknown moe_router mode"):
            MoERouterAlgorithm.validate_mode("supress")
