# SPDX-License-Identifier: Apache-2.0
"""Central algorithm -> CUDA execution mode mapping.

The table is derived from each algorithm class's declared graph_family
(single source of truth); graph_request_problem is the one admissibility
check behind graph_mode=full, shared by the frontend, the worker and the
auto graph-mode resolution.
"""

import numpy as np

from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
from vllm.steer_vectors.algorithms import steering_execution_modes
from vllm.steer_vectors.api import to_engine_request
from vllm.steer_vectors.payloads import LowRankProjector
from vllm.steer_vectors.graph_support import graph_request_problem

APPLY_ALL = ApplySpec(phases=["prompt", "generation"])


def _request(**vector_kwargs):
    vector_kwargs.setdefault("apply", APPLY_ALL)
    return to_engine_request(SteeringSpec(vectors=[VectorSpec(**vector_kwargs)]))


def _lowrank(rank):
    p = np.zeros((64, rank), dtype=np.float32)
    return LowRankProjector(p, p)


class TestExecutionModes:
    def test_table_matches_declared_families(self):
        modes = steering_execution_modes()
        assert modes["direct"] == ("split", "in_graph")
        assert modes["erase"] == ("split", "in_graph")
        assert modes["replace"] == ("split", "in_graph")
        assert modes["concept_replace"] == ("split", "in_graph")
        assert modes["loreft"] == ("split", "in_graph")
        assert modes["lm_steer"] == ("split", "in_graph")
        assert modes["moe_router"] == ("split", "in_graph")
        assert modes["linear"] == ("split",)

    def test_every_algorithm_supports_split(self):
        assert all("split" in m for m in steering_execution_modes().values())

    def test_conditional_classification(self):
        """auto resolves names pessimistically from this classification:
        conditional algorithms (rank-capped lowrank family, config-form
        moe_router) drop out of the unconditional set."""
        from vllm.steer_vectors.algorithms import (
            graph_condition,
            unconditionally_graph_safe_algorithms,
        )

        uncond = unconditionally_graph_safe_algorithms()
        assert uncond == {"concept_replace", "direct", "erase", "replace"}
        assert "rank" in graph_condition("lm_steer")
        assert "rank" in graph_condition("loreft")
        assert "inline" in graph_condition("moe_router")
        assert graph_condition("direct") is None
        assert graph_condition("linear") is None


class TestGraphRequestProblem:
    def test_graph_safe_config_passes(self):
        req = _request(source="v.gguf", algorithm="direct", scale=1.0,
                       layers=[10])
        assert graph_request_problem(req, max_rank=32) is None

    def test_ungraphable_algorithm_named(self):
        from vllm.steer_vectors.payloads import LinearMap

        weight = np.eye(8, dtype=np.float32)
        req = _request(data=LinearMap(weight), algorithm="linear",
                       scale=1.0, layers=[10])
        assert "linear" in graph_request_problem(req, max_rank=32)

    def test_moe_inline_toggle_admitted(self):
        req = _request(algorithm="moe_router", layers=[3],
                       params={"expert_ids": [1], "mode": "deactivate"})
        assert graph_request_problem(req, max_rank=32) is None

    def test_moe_soft_mode_rejected(self):
        req = _request(algorithm="moe_router", layers=[3],
                       params={"expert_ids": [1], "mode": "soft"})
        assert "soft" in graph_request_problem(req, max_rank=32)

    def test_moe_file_config_rejected(self):
        req = _request(source="moe.json", algorithm="moe_router")
        assert "file-based" in graph_request_problem(req, max_rank=32)

    def test_normalize_admitted(self):
        req = _request(source="v.gguf", algorithm="direct", scale=1.0,
                       layers=[10], normalize=True)
        assert graph_request_problem(req, max_rank=32) is None

    def test_multi_vector_rejected(self):
        spec = SteeringSpec(vectors=[
            VectorSpec(source="v.gguf", scale=1.0, layers=[10],
                       apply=APPLY_ALL),
            VectorSpec(source="v.gguf", scale=1.0, layers=[11],
                       apply=APPLY_ALL),
        ])
        problem = graph_request_problem(to_engine_request(spec), max_rank=32)
        assert "multi-vector" in problem

    def test_rank_within_limit_passes(self):
        req = _request(data=_lowrank(4), algorithm="lm_steer", scale=1.0,
                       layers=[10])
        assert graph_request_problem(req, max_rank=32) is None

    def test_rank_above_limit_rejected(self):
        req = _request(data=_lowrank(64), algorithm="lm_steer", scale=1.0,
                       layers=[10])
        assert "rank 64" in graph_request_problem(req, max_rank=32)
