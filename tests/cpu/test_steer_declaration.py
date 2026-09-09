# SPDX-License-Identifier: Apache-2.0
"""SteerVectorConfig workload-declaration validation.

The declaration (steer_algorithms) is the user-facing steering
contract: names are validated against the registry, "all" is exclusive,
and the retired graph-mode names point at their replacements. The
resolution ladder itself (auto -> in_graph/split) runs in VllmConfig
finalization and is covered end to end by the e2e engine modules.
"""

import json
from types import SimpleNamespace

import pytest
from vllm.config.steer_vector import SteerVectorConfig
from vllm.exceptions import VLLMClientError
from vllm.model_hooks.steering.request import ResolvedVector, SteeringRequest
from vllm.v1.engine.input_processor import InputProcessor


class TestAlgorithmsDeclaration:
    def test_list_is_normalized_sorted_deduped(self):
        cfg = SteerVectorConfig(algorithms=["lm_steer", "direct", "direct"])
        assert cfg.algorithms == ["direct", "lm_steer"]

    def test_all_cannot_mix_with_names(self):
        with pytest.raises(Exception, match="cannot be combined"):
            SteerVectorConfig(algorithms=["all", "direct"])

    def test_unknown_name_rejected_with_available_list(self):
        with pytest.raises(Exception, match="available"):
            SteerVectorConfig(algorithms=["direct", "does_not_exist"])

    def test_empty_declaration_rejected(self):
        with pytest.raises(Exception, match="must not be empty"):
            SteerVectorConfig(algorithms=[])

    def test_none_passes_validator(self):
        """None defers to VllmConfig finalization, which derives the
        declaration from steering_config or raises the declare-your-
        workload error."""
        assert SteerVectorConfig().algorithms is None


@pytest.mark.parametrize("algorithms", ["direct,lm_steer", "all"])
def test_cli_preserves_single_algorithm_value_and_dtype(algorithms):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.steer_vector_dtype == "auto" and defaults.max_steer_vectors is None
    args = parser.parse_args(
        [
            "--steer-algorithms",
            algorithms,
            "--steer-vector-dtype",
            "bfloat16",
        ]
    )
    assert args.steer_algorithms == algorithms
    config = SteerVectorConfig(
        algorithms=args.steer_algorithms,
        steer_vector_dtype=args.steer_vector_dtype,
    )
    assert config.algorithms == (
        "all" if algorithms == "all" else ["direct", "lm_steer"]
    )
    assert config.steer_vector_dtype == "bfloat16"


def test_config_rejects_extra_fields_and_preserves_internal_cache_defaults():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="unexpected_keyword_argument"):
        SteerVectorConfig(algorithms="direct", unused=True)
    config = SteerVectorConfig(algorithms="direct")
    assert config._default_request is None and config._default_request_spec is None
    config.resolve_workload(max_num_seqs=16, compiled=True)
    assert config.max_steer_vectors == 16 and config.graph_mode == "in_graph"


class TestGraphModeValues:
    def test_retired_names_point_at_replacements(self):
        with pytest.raises(Exception, match="in_graph"):
            SteerVectorConfig(graph_mode="full")
        with pytest.raises(Exception, match="split"):
            SteerVectorConfig(graph_mode="piecewise")

    def test_unknown_value_rejected(self):
        with pytest.raises(Exception, match="steer_graph_mode"):
            SteerVectorConfig(graph_mode="eager")

    def test_valid_values_accepted(self):
        for mode in ("auto", "in_graph", "split"):
            assert SteerVectorConfig(graph_mode=mode).graph_mode == mode


class TestCompilationSignature:
    def _hash(self, **overrides):
        config = {
            "algorithms": ["direct"],
            "graph_mode": "in_graph",
            "max_steer_vectors": 8,
            "steer_vector_dtype": "float32",
        }
        config.update(overrides)
        return SteerVectorConfig(**config).compute_hash()

    def test_declared_kernel_layout_changes_signature(self):
        direct = self._hash()
        for overrides in (
            {"algorithms": ["replace"]},
            {"algorithms": ["direct", "moe_router"]},
            {"max_steer_vectors": 16},
            {"steer_vector_dtype": "float16"},
            {"graph_mode": "split"},
        ):
            assert self._hash(**overrides) != direct
        assert self._hash(algorithms=["lm_steer"], graph_max_rank=4) != self._hash(
            algorithms=["lm_steer"], graph_max_rank=8
        )

    def test_same_family_and_unused_rank_share_signature(self):
        assert self._hash(algorithms=["erase"]) == self._hash(
            algorithms=["erase", "concept_replace"]
        )
        assert self._hash(graph_max_rank=4) == self._hash(graph_max_rank=64)

    def test_split_signature_tracks_component_hooks_only(self):
        decoder = self._hash(graph_mode="split")
        assert self._hash(graph_mode="split", algorithms=["loreft"]) == decoder
        assert self._hash(graph_mode="split", algorithms=["moe_router"]) != decoder
        assert self._hash(
            graph_mode="split", algorithms=["direct", "moe_router"]
        ) not in (decoder, self._hash(graph_mode="split", algorithms=["moe_router"]))

    def test_dynamic_default_payload_does_not_invalidate_compilation(self):
        first = {
            "vectors": [
                {
                    "source": "first.gguf",
                    "scale": 1.0,
                    "layers": [1],
                    "apply": {"generation": "all"},
                }
            ]
        }
        second = {
            "vectors": [
                {
                    "source": "second.gguf",
                    "scale": 2.0,
                    "layers": [2],
                    "normalize": True,
                    "apply": {"prompt_positions": [-1]},
                }
            ]
        }
        assert self._hash(steering_config=json.dumps(first)) == self._hash(
            steering_config=json.dumps(second)
        )


def _admission_processor(**config_overrides):
    """Exercise real admission without constructing a tokenizer or an engine."""
    config = {
        "algorithms": ["direct"],
        "multi_vector": False,
        "require_preload": False,
        "graph_mode": "split",
        "graph_max_rank": 32,
    }
    config.update(config_overrides)
    processor = object.__new__(InputProcessor)
    processor.vllm_config = SimpleNamespace(
        steer_vector_config=SimpleNamespace(**config), kv_transfer_config=None
    )
    processor._steer_preloaded_paths = set()
    processor._steer_preloaded_payloads = set()
    processor.model_config = SimpleNamespace(get_hidden_size=lambda: 2)
    processor._steering_model_info = {
        "hidden_states": {0: 2, 1: 2},
        "router_logits": {1: 4},
    }
    return processor


def _direct_request():
    from vllm.model_hooks.steering.api import (
        ApplySpec,
        SteeringSpec,
        VectorSpec,
        to_engine_request,
    )
    from vllm.model_hooks.steering.payloads import DirectionVector

    return to_engine_request(
        SteeringSpec(
            vectors=[
                VectorSpec(
                    data=DirectionVector({1: [1.0, 2.0]}),
                    apply=ApplySpec(generation="all"),
                )
            ]
        )
    )


class TestAdmissionErrorTypes:
    """0.28 AsyncLLM propagates VLLMClientError as a request error (4xx)."""

    @pytest.mark.parametrize("problem", ["width", "layer", "partial_vector", "topk"])
    def test_model_errors_rejected_without_worker_admission(self, problem):
        from vllm.model_hooks.steering.api import to_engine_request
        from vllm.steer_vectors import (
            ApplySpec,
            DirectionVector,
            SteeringSpec,
            VectorSpec,
        )

        processor = _admission_processor(algorithms="all", multi_vector=True)
        if problem == "topk":
            vectors = [
                VectorSpec(
                    algorithm="moe_router",
                    layers=[1],
                    params={"expert_ids": [1], "mode": "soft_topk", "topk": 5},
                    apply=ApplySpec(generation="all"),
                )
            ]
            match = "topk.*exceeds expert count"
        else:
            vectors = [
                VectorSpec(
                    data=DirectionVector(
                        {
                            9 if problem != "width" else 1: [1.0, 2.0, 3.0]
                            if problem == "width"
                            else [1.0, 2.0]
                        }
                    ),
                    apply=ApplySpec(generation="all"),
                )
            ]
            if problem == "partial_vector":
                vectors.insert(
                    0,
                    VectorSpec(
                        data=DirectionVector({1: [1.0, 2.0]}),
                        apply=ApplySpec(generation="all"),
                    ),
                )
            match = "hidden size" if problem == "width" else "targets no modules"
        with pytest.raises(VLLMClientError, match=match):
            processor._validate_steer_vector(
                to_engine_request(SteeringSpec(vectors=vectors))
            )
        # A rejected configuration does not poison subsequent admission.
        processor._validate_steer_vector(_direct_request())

    def test_prompt_overlap_uses_tokens_and_actual_target_layers(self):
        from vllm.model_hooks.steering.api import to_engine_request
        from vllm.model_hooks.steering.validation import validate_prompt_conflicts
        from vllm.steer_vectors import (
            ApplySpec,
            DirectionVector,
            SteeringSpec,
            VectorSpec,
        )

        first = VectorSpec(
            data=DirectionVector({0: [1.0, 2.0]}), apply=ApplySpec(prompt_tokens=[7])
        )
        second = VectorSpec(
            data=DirectionVector({0: [1.0, 2.0]}),
            apply=ApplySpec(prompt_positions=[-1]),
        )
        request = to_engine_request(
            SteeringSpec(vectors=[first, second], conflict="error")
        )
        validate_prompt_conflicts(request, [7, 8])
        with pytest.raises(ValueError, match="conflict at prompt positions"):
            validate_prompt_conflicts(request, [8, 7])
        second.data = DirectionVector({1: [1.0, 2.0]})
        request = to_engine_request(
            SteeringSpec(vectors=[first, second], conflict="error")
        )
        validate_prompt_conflicts(request, [8, 7])

    def test_random_router_rejected_before_execution(self):
        from vllm.model_hooks.steering.api import (
            ApplySpec,
            SteeringSpec,
            VectorSpec,
            to_engine_request,
        )

        processor = _admission_processor(
            algorithms=["moe_router"], graph_mode="in_graph"
        )
        request = to_engine_request(
            SteeringSpec(
                vectors=[
                    VectorSpec(
                        source=None,
                        algorithm="moe_router",
                        layers=[0],
                        params={"expert_ids": [1], "mode": "soft_random"},
                        apply=ApplySpec(generation="all"),
                    )
                ]
            )
        )
        with pytest.raises(VLLMClientError, match="soft_random.*split"):
            processor._validate_steer_vector(request)

    def test_in_graph_rejection_keeps_its_configuration_guidance(self):
        processor = _admission_processor(multi_vector=True, graph_mode="in_graph")
        request = SteeringRequest(
            "graph-admission-test",
            1,
            vectors=[
                ResolvedVector(
                    source=path,
                    apply_spec={"generation": "all"},
                    payload=_direct_request().vectors[0].payload,
                )
                for path in ("/not-loaded/a.gguf", "/not-loaded/b.gguf")
            ],
        )
        with pytest.raises(VLLMClientError, match="multi-vector configs.*split"):
            processor._validate_steer_vector(request)

    @pytest.mark.parametrize(
        ("wire", "reason"),
        [
            ({"phases": "generation"}, "unknown selection fields"),
            ({"generation_positions": [-1]}, "0-based decode steps"),
            ({"prompt_window": 5}, "not iterable"),
        ],
    )
    def test_capture_wire_errors_are_client_errors_with_stream_context(
        self, wire, reason
    ):
        from vllm.model_hooks.capture.policy import normalize_capture_select

        with pytest.raises(VLLMClientError) as caught:
            normalize_capture_select({"hidden": wire})
        assert "capture_select['hidden']" in str(caught.value)
        assert reason in str(caught.value)
        assert caught.value.__cause__ is not None


class TestDefaultRequestSnapshot:
    def test_parsed_default_survives_source_changes_pickle_and_deepcopy(self, tmp_path):
        import copy
        import pickle

        from vllm.model_hooks.steering.defaults import build_default_request
        from vllm.model_hooks.steering.request import config_fingerprint

        path = tmp_path / "router.json"
        path.write_text(
            json.dumps(
                {
                    "layer_configs": {
                        "1": {"expert_ids": [1], "mode": "activate"},
                    }
                }
            )
        )
        spec = {
            "vectors": [
                {
                    "source": str(path),
                    "algorithm": "moe_router",
                    "apply": {"generation": "all"},
                }
            ]
        }
        config = SteerVectorConfig(
            algorithms=["moe_router"],
            graph_mode="in_graph",
            steering_config=json.dumps(spec),
        )
        signature = config.compute_hash()
        original = build_default_request(config)
        fingerprint = config_fingerprint(original)
        path.write_text(
            json.dumps(
                {
                    "layer_configs": {
                        "1": {"expert_ids": [2], "mode": "soft"},
                    }
                }
            )
        )
        copies = (config, copy.deepcopy(config), pickle.loads(pickle.dumps(config)))
        for copied in copies:
            request = build_default_request(copied)
            assert request.vectors[0].payload == original.vectors[0].payload
            assert config_fingerprint(request) == fingerprint
            assert copied.compute_hash() == signature
        fresh = SteerVectorConfig(
            algorithms=["moe_router"],
            graph_mode="in_graph",
            steering_config=json.dumps(spec),
        )
        assert (
            build_default_request(fresh)
            .vectors[0]
            .payload["extra"]["layers"]["1"]["mode"]
            == "soft"
        )
        assert fresh.compute_hash() == signature
        # A changed spec invalidates the old snapshot even on the same config.
        spec["vectors"][0]["params"] = {"mode": "deactivate"}
        config.steering_config = json.dumps(spec)
        replaced = build_default_request(config)
        assert (
            replaced.vectors[0].payload["extra"]["layers"]["1"]["mode"] == "deactivate"
        )
        assert replaced is not original
