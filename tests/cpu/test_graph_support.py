# SPDX-License-Identifier: Apache-2.0
"""Central algorithm -> CUDA execution mode mapping.

The table is derived from each algorithm class's declared graph_family
(single source of truth); graph_request_problem is the one admissibility
check behind graph_mode=in_graph, shared by the frontend, the worker and the
auto graph-mode resolution.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.model_hooks.steering.api import to_engine_request
from vllm.model_hooks.steering.graph.policy import (
    graph_request_problem,
    steering_execution_modes,
)
from vllm.model_hooks.steering.payloads import DirectionVector, LowRankProjector
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

APPLY_ALL = ApplySpec(prompt="all", generation="all")


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

    def test_conditional_classification(self):
        """Names alone cannot promise a bounded rank or nonrandom router mode."""
        from vllm.model_hooks.steering.graph.policy import graph_problem

        assert graph_problem("direct") is None
        assert "rank" in graph_problem("lm_steer")
        assert "rank" in graph_problem("loreft")
        assert "soft_random" in graph_problem("moe_router")
        assert "no in-graph kernel" in graph_problem("linear")


class TestDeclaredGraphFamilies:
    """The kernel is compiled with exactly the declared workload's
    families (declared_graph_families); moe_router maps to the gate
    kernel, not a decoder family."""

    def test_family_mapping(self):
        from vllm.model_hooks.steering.graph.policy import declared_graph_families

        assert declared_graph_families(["direct"]) == {"additive"}
        assert declared_graph_families(["direct", "erase"]) == {
            "additive", "projection",
        }
        assert declared_graph_families(["lm_steer"]) == {"lowrank"}
        assert declared_graph_families(["replace"]) == {"replace"}
        assert declared_graph_families(["moe_router"]) == frozenset()
        assert declared_graph_families("all") == {
            "additive", "projection", "lowrank", "replace",
        }
        assert declared_graph_families(None) == {
            "additive", "projection", "lowrank", "replace",
        }


class TestGraphMaskStorage:
    """Only masks read by declared kernels occupy the shared step buffer."""

    def _state(self, algorithms):
        import torch
        from vllm.config.steer_vector import SteerVectorConfig
        from vllm.model_hooks.steering.controllers import (
            HiddenStatesController,
            RouterLogitsController,
        )
        from vllm.model_hooks.steering.graph.state import SteeringGraphState

        decoders = [HiddenStatesController() for _ in range(2)]
        gate = RouterLogitsController()
        gate.hook_target = torch.nn.Linear(8, 4, bias=False)
        state = SteeringGraphState(
            SteerVectorConfig(algorithms=algorithms, max_steer_vectors=2),
            torch.device("cpu"),
        )
        state.enable(8, torch.float32, 16)
        controllers = {"decoder0": decoders[0], "decoder1": decoders[1]}
        if "moe_router" in algorithms:
            controllers["gate"] = gate
        state.init_tables(SimpleNamespace(controllers=controllers))
        return state, decoders, gate

    def test_direct_has_no_replace_or_gate_mask(self):
        state, decoders, gate = self._state(["direct"])
        assert state.step_masks.shape == (2, 16)
        assert all(decoder.replace_mask is None for decoder in decoders)
        assert gate.graph_mask is None and gate.graph_tables is None
        state.step_masks.fill_(1)
        state.zero_step_masks(5)
        for decoder in decoders:
            assert decoder.graph_mask[:5].count_nonzero() == 0
            assert (decoder.graph_mask[5:] == 1).all()
        # A larger later replay must clear its whole padded span too.
        state.zero_step_masks(12)
        assert state.step_masks[:, :12].count_nonzero() == 0
        assert (state.step_masks[:, 12:] == 1).all()

    def test_moe_only_does_not_allocate_decoder_buffers(self):
        state, decoders, gate = self._state(["moe_router"])
        assert state.controllers == [gate]
        assert state.step_masks.shape == (1, 16)
        for decoder in decoders:
            assert decoder._graph_mode and decoder.graph_tables == {}
            assert decoder.graph_mask is None
            assert decoder.replace_mask is None
            assert decoder.normalize_flag is None

    def test_replace_only_allocates_its_own_mask(self):
        state, decoders, _ = self._state(["replace"])
        assert state.step_masks.shape == (2, 16)
        for decoder in decoders:
            assert decoder.graph_mask is None
            assert decoder.replace_mask is not None

    def test_normalize_has_distinct_identity_and_persistent_row_state(self):
        import torch
        from vllm.model_hooks.steering.payloads import DirectionVector
        from vllm.model_hooks.steering.request import config_fingerprint

        payload = DirectionVector({0: np.ones(8, dtype=np.float32)})
        plain, normalized = [
            _request(data=payload, layers=[0], normalize=flag)
            for flag in (False, True)
        ]
        assert graph_request_problem(plain, max_rank=32) is None
        assert graph_request_problem(normalized, max_rank=32) is None
        assert config_fingerprint(plain) != config_fingerprint(normalized)
        state, decoders, _ = self._state(["direct"])
        manager = SimpleNamespace(
            controllers_for_layer=lambda layer, kind: [decoders[layer]]
        )
        layer_payloads = {0: torch.ones(8)}
        # Install the upper slot first: its row must not depend on admission order.
        state.distribute(1, normalized.vectors[0], layer_payloads, manager)
        assert state.row_of(0) == 0
        assert state.row_of(1) == 2
        state.distribute(0, plain.vectors[0], layer_payloads, manager)
        plain_row, normalized_row = state.row_of(0), state.row_of(1)
        assert (plain_row, normalized_row) == (1, 2)
        assert decoders[0].normalize_flag[plain_row] == 0
        assert decoders[0].normalize_flag[normalized_row] == 1
        assert decoders[1].normalize_flag.count_nonzero() == 0
        state.release(1)
        assert state.row_of(1) == 0
        assert decoders[0].normalize_flag[normalized_row] == 0
        state.distribute(1, plain.vectors[0], layer_payloads, manager)
        assert state.row_of(1) == normalized_row
        assert decoders[0].normalize_flag[normalized_row] == 0


class TestFamilySpecializedKernel:
    """A kernel compiled with a family subset must match the full
    kernel whenever the dropped families are idle (their tables and
    masks all-zero) — the exactness the specialization relies on."""

    def _buffers(self, rows=4, hidden=8, rank=2, n=6):
        import torch
        from vllm.model_hooks.steering.graph.kernels import GRAPH_FAMILIES

        g = torch.Generator().manual_seed(0)
        dim_of = {"h": hidden, "r": rank}
        tables = {
            family: {
                key: torch.zeros(rows, *(dim_of[d] for d in dims))
                for key, dims in schema.items()
            }
            for family, schema in GRAPH_FAMILIES.items()
        }
        tables["additive"]["V"][1:] = torch.randn(
            rows - 1, hidden, generator=g
        )
        hidden_states = torch.randn(n, hidden, generator=g)
        residual = torch.randn(n, hidden, generator=g)
        graph_mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        replace_mask = torch.zeros(n)
        normalize_flag = torch.zeros(rows)
        normalize_flag[2] = 1.0
        token_rows = torch.tensor([1, 0, 2, 3, 0, 2])
        return (tables, graph_mask, replace_mask, normalize_flag,
                token_rows, hidden_states, residual)

    def test_additive_subset_matches_full(self):
        from vllm.model_hooks.steering.graph.kernels import apply_decoder_families

        (tables, graph_mask, replace_mask, normalize_flag, token_rows,
         hidden_states, residual) = self._buffers()
        full = apply_decoder_families(
            tables, graph_mask, replace_mask, normalize_flag, token_rows,
            hidden_states, residual,
        )
        subset = apply_decoder_families(
            {"additive": tables["additive"]}, graph_mask, replace_mask,
            normalize_flag, token_rows, hidden_states, residual,
        )
        assert (full == subset).all(), (
            "additive-only kernel differs from full kernel with idle "
            "families"
        )

    def test_empty_families_is_identity(self):
        from vllm.model_hooks.steering.graph.kernels import apply_decoder_families

        (_, graph_mask, replace_mask, normalize_flag, token_rows,
         hidden_states, residual) = self._buffers()
        out = apply_decoder_families(
            {}, graph_mask, replace_mask, normalize_flag, token_rows,
            hidden_states, residual,
        )
        assert out is hidden_states


class TestGraphRequestProblem:
    def test_ungraphable_algorithm_named(self):
        from vllm.model_hooks.steering.payloads import LinearMap

        weight = np.eye(8, dtype=np.float32)
        req = _request(data=LinearMap(weight), algorithm="linear",
                       scale=1.0, layers=[10])
        assert "linear" in graph_request_problem(req, max_rank=32)

    def test_moe_inline_toggle_admitted(self):
        req = _request(algorithm="moe_router", layers=[3],
                       params={"expert_ids": [1], "mode": "deactivate"})
        assert graph_request_problem(req, max_rank=32) is None

    def test_moe_soft_mode_admitted(self):
        req = _request(algorithm="moe_router", layers=[3],
                       params={"expert_ids": [1], "mode": "soft"})
        assert graph_request_problem(req, max_rank=32) is None

    def test_moe_random_mode_rejected(self):
        req = _request(algorithm="moe_router", layers=[3],
                       params={"expert_ids": [1], "mode": "soft_random"})
        assert "soft_random" in graph_request_problem(req, max_rank=32)

    def test_multi_vector_rejected(self):
        spec = SteeringSpec(vectors=[
            VectorSpec(data=DirectionVector({10: np.ones(8), 11: np.ones(8)}), scale=1.0, layers=[10],
                       apply=APPLY_ALL),
            VectorSpec(data=DirectionVector({10: np.ones(8), 11: np.ones(8)}), scale=1.0, layers=[11],
                       apply=APPLY_ALL),
        ])
        problem = graph_request_problem(to_engine_request(spec), max_rank=32)
        assert "multi-vector" in problem

    def test_rank_within_limit_passes(self):
        req = _request(data=_lowrank(4), algorithm="lm_steer", scale=1.0,
                       layers=[10])
        assert graph_request_problem(req, max_rank=32) is None

class TestGraphModeResolution:
    """Boot decisions and request admission consume the same capability check."""

    @pytest.mark.parametrize(
        "algorithm", ["direct", "lm_steer", "moe_router", "linear"]
    )
    def test_names_only_auto_uses_the_same_condition(self, algorithm):
        from vllm.config.steer_vector import SteerVectorConfig
        from vllm.model_hooks.steering.graph.policy import (
            graph_problem,
            resolve_graph_mode,
        )

        config = SteerVectorConfig(algorithms=[algorithm])
        mode, reason = resolve_graph_mode(config, compiled=True)
        problem = graph_problem(algorithm)
        assert mode == ("split" if problem else "in_graph")
        if problem:
            assert problem in reason

    def test_exact_lowrank_payload_can_resolve_in_graph(self):
        from vllm.config.steer_vector import SteerVectorConfig
        from vllm.model_hooks.steering.graph.policy import resolve_graph_mode

        config = SteerVectorConfig(algorithms=["lm_steer"], graph_max_rank=4)
        for rank in (4, 8):
            request = _request(data=_lowrank(rank), algorithm="lm_steer", layers=[1])
            mode, reason = resolve_graph_mode(
                config, compiled=True, default_request=request
            )
            problem = graph_request_problem(request, max_rank=4)
            if rank == 4:
                assert problem is None and mode == "in_graph"
            else:
                assert "rank 8" in problem and mode == "split"
                assert problem in reason
                config.graph_mode = "in_graph"
                with pytest.raises(ValueError, match="rank 8"):
                    resolve_graph_mode(config, compiled=True, default_request=request)

    def test_explicit_mode_never_falls_back(self):
        from vllm.config.steer_vector import SteerVectorConfig
        from vllm.model_hooks.steering.graph.policy import resolve_graph_mode

        for kwargs, pattern in (
            ({"algorithms": ["linear"]}, "no in-graph kernel"),
            ({"algorithms": ["direct"], "multi_vector": True}, "multi-vector"),
            ({"algorithms": "all"}, "steer_algorithms='all'"),
        ):
            config = SteerVectorConfig(graph_mode="in_graph", **kwargs)
            with pytest.raises(ValueError, match=pattern):
                resolve_graph_mode(config, compiled=True)
        config = SteerVectorConfig(algorithms=["direct"], graph_mode="in_graph")
        with pytest.raises(ValueError, match="requires compiled"):
            resolve_graph_mode(config, compiled=False)
        config.graph_mode = "auto"
        assert resolve_graph_mode(config, compiled=False)[0] == "split"


class TestRouterGraphPayloads:
    def test_transposed_gate_weight_uses_logit_width_for_graph_tables(self):
        import torch
        from vllm.model_hooks.steering.controllers import RouterLogitsController
        from vllm.model_hooks.steering.graph.kernels import apply_gate_intervention

        # FP8 linear loading stores weight as [hidden, experts]. Logical
        # output_size remains the expert count consumed by the routing kernel.
        gate = torch.nn.Module()
        gate.output_size = 4
        gate.weight = torch.nn.Parameter(torch.empty(8, 4), requires_grad=False)
        controller = RouterLogitsController()
        controller.hook_target = gate
        rows = torch.tensor([1, 0, 1])
        mask = torch.tensor([1.0, 1.0, 0.0])
        controller.init_graph_buffers(
            frozenset({"moe_gate"}),
            {"gate": mask},
            capacity=2,
            hidden_size=8,
            max_rank=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
            token_rows=rows,
        )
        assert controller.output_width == 4
        assert controller.graph_tables["activate"].shape == (3, 4)
        controller.set_graph_row(
            1,
            "moe_router",
            {
                "mode": "activate",
                "expert_ids": [1],
                "deactivate_ids": [3],
                "epsilon": 0.25,
            },
            1.0,
        )
        logits = torch.arange(4, dtype=torch.float32).expand(3, -1).clone()
        expected = logits.clone()
        scores = torch.log_softmax(logits[0], dim=-1)
        expected[0] = scores
        expected[0, 1] = scores.max() + 0.25
        expected[0, 3] = scores.min() - 0.25
        actual = logits.clone()
        apply_gate_intervention(controller.graph_tables, mask, rows, actual)
        torch.testing.assert_close(actual, expected)
        controller.clear_graph_row(1)
        actual.copy_(logits)
        apply_gate_intervention(controller.graph_tables, mask, rows, actual)
        assert torch.equal(actual, logits)
    def test_file_snapshot_and_inline_share_admission_and_payload(self, tmp_path):
        import msgspec
        from vllm.model_hooks.steering.request import (
            SteeringRequest,
            config_fingerprint,
        )

        path = tmp_path / "router.json"
        path.write_text(json.dumps({"layer_configs": {
            "1": {"expert_ids": [1, 3], "mode": "soft_topk",
                  "lambda": -0.75, "topk": 2},
            "2": {"expert_ids": [2], "mode": "soft_random"},
        }}))
        request = _request(source=str(path), algorithm="moe_router", layers=[1])
        inline = _request(algorithm="moe_router", layers=[1], params={
            "expert_ids": [1, 3], "mode": "soft_topk", "lambda": -0.75, "topk": 2,
        })
        assert request.vectors[0].payload["extra"]["layers"]["1"] == inline.vectors[0].payload["extra"]["layers"]["1"]
        assert graph_request_problem(request, 32) is None
        fingerprint = config_fingerprint(request)
        encoded = msgspec.msgpack.encode(request)
        # Worker decoding must use the admitted snapshot, even if the file changes.
        path.write_text(json.dumps({"layer_configs": {
            "1": {"expert_ids": [2], "mode": "soft_random"},
        }}))
        decoded = msgspec.msgpack.decode(encoded, type=SteeringRequest)
        assert decoded.vectors[0].payload["extra"]["layers"]["1"] == inline.vectors[0].payload["extra"]["layers"]["1"]
        assert graph_request_problem(decoded, 32) is None
        assert config_fingerprint(decoded) == fingerprint
        changed = _request(source=str(path), algorithm="moe_router", layers=[1])
        assert "soft_random" in graph_request_problem(changed, 32)
        assert config_fingerprint(changed) != fingerprint

    def test_request_mode_override_and_selected_layers_are_resolved(self, tmp_path):
        path = tmp_path / "router.json"
        path.write_text(json.dumps({"layer_configs": {
            "1": {"expert_ids": [1], "mode": "soft_random"},
            "2": {"expert_ids": [2], "mode": "soft"},
        }}))
        request = _request(source=str(path), algorithm="moe_router", layers=[2],
                           params={"mode": "deactivate"})
        assert request.vectors[0].payload["extra"]["layers"]["2"]["mode"] == "deactivate"
        assert graph_request_problem(request, 32) is None

    def test_router_tables_match_eager_and_clear_reused_rows(self):
        import torch
        from vllm.model_hooks.steering.algorithms.moe_router import MoERouterAlgorithm
        from vllm.model_hooks.steering.controllers import RouterLogitsController
        from vllm.model_hooks.steering.graph.kernels import apply_gate_intervention

        controller = RouterLogitsController()
        controller.hook_target = torch.nn.Linear(4, 8, bias=False)
        rows = torch.tensor([1, 2, 3, 4, 0, 1])
        mask = torch.tensor([1., 1., 1., 1., 1., 0.])
        controller.init_graph_buffers(
            frozenset({"moe_gate"}), {"gate": mask}, capacity=4,
            hidden_size=4, max_rank=2, dtype=torch.float32,
            device=torch.device("cpu"), token_rows=rows,
        )
        payloads = [
            {"mode": "activate", "expert_ids": [1, 3], "deactivate_ids": [3]},
            {"mode": "deactivate", "expert_ids": [5], "activate_ids": [1]},
            {"mode": "soft", "expert_ids": [1, 5], "lambda": -0.75},
            {"mode": "soft_topk", "expert_ids": [1, 7], "lambda": 0.75, "topk": 2},
        ]
        logits = torch.arange(8, dtype=torch.float32).expand(6, -1).clone()
        expected = logits.clone()
        algo = MoERouterAlgorithm()
        for row, payload in enumerate(payloads, 1):
            controller.set_graph_row(row, "moe_router", payload, 1.0)
            expected[row - 1] = algo._transform(logits[row - 1:row], payload)[0]
        actual = logits.clone()
        apply_gate_intervention(controller.graph_tables, mask, rows, actual)
        torch.testing.assert_close(actual, expected)
        assert torch.equal(actual[4:], logits[4:])
        controller.clear_graph_row(4)
        actual.copy_(logits)
        apply_gate_intervention(controller.graph_tables, mask, rows, actual)
        assert torch.equal(actual[3], logits[3])
        with pytest.raises(ValueError, match="exceeds expert count"):
            controller.set_graph_row(4, "moe_router", {
                "mode": "soft_topk", "expert_ids": [1], "topk": 9,
            }, 1.0)


class TestGraphAdmissionAllocation:
    @pytest.mark.parametrize("mode", ["split", "in_graph"])
    def test_remote_pipeline_targets_use_no_local_interventions_or_graph_rows(self, mode):
        import torch
        from vllm.config.steer_vector import SteerVectorConfig
        from vllm.model_hooks.steering.worker_manager import WorkerSteeringState

        config = SteerVectorConfig(
            algorithms=["moe_router"], max_steer_vectors=2, graph_mode=mode,
            steer_vector_dtype="float32",
        )
        worker = WorkerSteeringState(torch.device("cpu"), config, hidden_size=8)
        worker._controller_manager = SimpleNamespace(
            controllers={}, controllers_for_layer=lambda layer, kind: [],
        )
        request = _request(algorithm="moe_router", layers=[1],
                           params={"expert_ids": [1]})
        # Global admission accepts this layer on another pipeline stage.
        slot = worker.acquire_config("remote", request)
        assert worker.slot_for_request("remote") == slot
        assert worker.graph_state.row_of(slot) == 0
        assert worker.graph_state.slot_controllers == {}
        assert worker.graph_batch_entries() == {}
        worker.release_config("remote")
        assert worker._config_slots == {}
        assert not worker.slot_clauses()

    @pytest.mark.parametrize("width", [1, 9])
    def test_graph_table_write_error_releases_the_partial_row(self, width):
        import torch
        from vllm.model_hooks.steering.payloads import DirectionVector

        state, decoders, _ = TestGraphMaskStorage()._state(["direct"])
        manager = SimpleNamespace(
            controllers_for_layer=lambda layer, kind: [decoders[layer]],
        )
        request = _request(data=DirectionVector({0: np.ones(8), 1: np.ones(8)}))
        with pytest.raises(ValueError, match="hidden dimensions must match exactly"):
            state.distribute(
                1, request.vectors[0], {0: torch.ones(8), 1: torch.ones(width)}, manager
            )
        assert state.row_of(1) == 0
        assert state.slot_controllers == {}
        for decoder in decoders:
            assert decoder.normalize_flag.count_nonzero() == 0
            assert decoder.graph_tables["additive"]["V"].count_nonzero() == 0
        state.distribute(1, request.vectors[0], {0: torch.ones(8)}, manager)
        assert state.row_of(1) == 2
        assert torch.equal(decoders[0].graph_tables["additive"]["V"][2], torch.ones(8))

    def test_graph_pads_only_lowrank_axes(self):
        import torch

        _, decoders, _ = TestGraphMaskStorage()._state(["lm_steer"])
        decoder = decoders[0]
        payload = {"projector1": torch.ones(8, 2), "projector2": torch.ones(8, 2)}
        decoder.set_graph_row(1, "lm_steer", payload, 0.5)
        tables = decoder.graph_tables["lowrank"]
        assert torch.equal(tables["A"][1, :, :2], payload["projector1"])
        assert torch.equal(tables["Rout"][1, :, :2], payload["projector2"] * 0.5)
        assert tables["A"][1, :, 2:].count_nonzero() == 0
        assert tables["Rout"][1, :, 2:].count_nonzero() == 0
        assert tables["b"][1].count_nonzero() == 0

class TestRequestInstallationTransaction:
    @pytest.mark.parametrize("mode", ["split", "in_graph"])
    def test_failed_request_leaves_existing_request_and_capacity_intact(self, mode):
        import torch
        from vllm.config.steer_vector import SteerVectorConfig
        from vllm.model_hooks.steering.controllers import HiddenStatesController
        from vllm.model_hooks.steering.worker_manager import WorkerSteeringState

        worker = WorkerSteeringState(
            torch.device("cpu"),
            SteerVectorConfig(
                algorithms=["direct"], max_steer_vectors=2, graph_mode=mode,
                steer_vector_dtype="float32",
            ),
            hidden_size=4,
        )
        decoder = HiddenStatesController()
        decoder.layer_id = 0
        worker._controller_manager = SimpleNamespace(
            controllers={"layer0": decoder},
            controllers_for_layer=lambda layer, kind: [decoder] if layer == 0 else [],
        )
        if mode == "in_graph":
            worker.enable_graph_mode(4, torch.float32, 8)
            worker.graph_state.init_tables(worker._controller_manager)
        original = _request(data=DirectionVector({0: np.ones(4)}), normalize=True)
        slot = worker.acquire_config("active", original)
        invalid = _request(data=DirectionVector({0: np.ones(5)}))
        with pytest.raises(ValueError, match="model hidden size 4"):
            worker.acquire_config("invalid", invalid)
        assert worker.slot_for_request("active") == slot
        assert worker.slot_for_request("invalid") is None
        assert len(worker._config_slots) == 1
        if mode == "in_graph":
            assert worker.graph_state.row_of(slot) == slot + 1
            assert decoder.normalize_flag[worker.graph_state.row_of(slot)] == 1
        else:
            assert decoder.slot_interventions[slot][0].normalize
        valid = _request(data=DirectionVector({0: np.full(4, 2.)}))
        assert worker.acquire_config("second", valid) != slot
        worker.release_config("active")
        worker.release_config("second")
        assert not worker.slot_clauses()
