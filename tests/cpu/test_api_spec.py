# SPDX-License-Identifier: Apache-2.0
"""CPU units for the v2 steering API (STEERING_API_V2.md).

Spec validation (explicit-failure paths), translation to the internal
engine struct, fingerprint/prefix-cache participation, and the position
collector's semantics: exclusions always compose, exact half-open
generation windows, negative positions resolved from the prompt end.
"""

import pytest
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

VEC = "/tmp/does-not-need-to-exist.gguf"
APPLY_ALL = ApplySpec(phases=["prompt", "generation"])


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


class TestApplySpecValidation:
    @pytest.mark.parametrize(
        "kwargs, needle",
        [
            ({"phases": []}, "non-empty"),
            ({"phases": ["prompt", "prompt"]}, "duplicates"),
            ({"phases": ["prefill"]}, None),
            ({"phases": ["prompt"], "tokens": [-1]}, "real token ids"),
            ({"phases": ["prompt"], "generation_window": (0, 3)}, "generation"),
            ({"phases": ["generation"], "generation_window": (3, 3)}, "half-open"),
            ({"phases": ["prompt"], "tokens": []}, "non-empty"),
        ],
    )
    def test_invalid_specs_rejected(self, kwargs, needle):
        with pytest.raises(ValidationError, match=needle):
            ApplySpec(**kwargs)

    def test_unbounded_window_accepted(self):
        assert ApplySpec(phases=["generation"], generation_window=(2, None))


class TestVectorAndSteeringSpecValidation:
    def test_path_algo_hack_rejected(self):
        with pytest.raises(ValidationError, match="plain path"):
            VectorSpec(source="v.gguf|linear", apply=APPLY_ALL)

    def test_unknown_params_rejected(self):
        with pytest.raises(ValidationError, match="unknown params"):
            VectorSpec(source=VEC, params={"beta": 1}, apply=APPLY_ALL)

    def test_source_required_for_non_moe(self):
        with pytest.raises(ValidationError, match="requires a source"):
            VectorSpec(algorithm="direct", apply=APPLY_ALL)

    def test_moe_requires_expert_ids_and_layers(self):
        with pytest.raises(ValidationError, match="expert_ids"):
            VectorSpec(algorithm="moe_router", layers=[3], apply=APPLY_ALL)
        with pytest.raises(ValidationError, match="layers"):
            VectorSpec(
                algorithm="moe_router",
                params={"expert_ids": [1]},
                apply=APPLY_ALL,
            )

    def test_empty_vectors_rejected(self):
        with pytest.raises(ValidationError, match="non-empty"):
            SteeringSpec(vectors=[])

    def test_multi_vector_moe_rejected(self):
        with pytest.raises(ValidationError, match="moe_router"):
            SteeringSpec(
                vectors=[
                    VectorSpec(source=VEC, apply=APPLY_ALL),
                    VectorSpec(
                        algorithm="moe_router",
                        layers=[3],
                        params={"expert_ids": [1]},
                        apply=APPLY_ALL,
                    ),
                ]
            )


class TestTranslation:
    def test_single_vector_fields(self):
        spec = make_spec(phases=["prompt"], exclude_positions=[0], tokens=[5])
        req = to_engine_request(spec)
        assert req.steer_vector_local_path == VEC
        assert req.scale == 0.5 and req.target_layers == [10]
        assert req.apply_spec == {
            "phases": ["prompt"],
            "tokens": [5],
            "positions": None,
            "exclude_tokens": None,
            "exclude_positions": [0],
            "window": None,
        }

    def test_moe_params_folded(self):
        req = to_engine_request(
            SteeringSpec(
                vectors=[
                    VectorSpec(
                        algorithm="moe_router",
                        layers=[3],
                        params={
                            "expert_ids": [1, 2],
                            "mode": "soft",
                            "lambda": 0.7,
                        },
                        apply=APPLY_ALL,
                    )
                ]
            )
        )
        assert req.moe_expert_ids == [1, 2]
        assert req.moe_mode == "soft"
        assert req.moe_lambda == 0.7
        assert req.moe_topk == 8

    def test_multi_vector_translation(self):
        req = to_engine_request(
            SteeringSpec(
                vectors=[
                    VectorSpec(source=VEC, layers=[10], apply=APPLY_ALL),
                    VectorSpec(
                        source=VEC,
                        layers=[12],
                        apply=ApplySpec(phases=["generation"]),
                    ),
                ],
                conflict="sequential",
            )
        )
        assert req.is_multi_vector and len(req.vector_configs) == 2
        assert req.conflict_resolution == "sequential"
        assert req.vector_configs[1].apply_spec["phases"] == ["generation"]


class TestEngineStructValidation:
    def test_missing_apply_spec_rejected(self):
        with pytest.raises(ValueError, match="apply_spec"):
            SteerVectorRequest(
                steer_vector_name="x",
                steer_vector_int_id=7,
                steer_vector_local_path=VEC,
            )

    def test_malformed_apply_spec_rejected(self):
        with pytest.raises(ValueError, match="unknown keys"):
            SteerVectorRequest(
                steer_vector_name="x",
                steer_vector_int_id=7,
                steer_vector_local_path=VEC,
                apply_spec={"phases": ["prompt"], "bogus": 1},
            )

    def test_apply_spec_satisfies_trigger_requirement(self):
        assert SteerVectorRequest(
            steer_vector_name="x",
            steer_vector_int_id=7,
            steer_vector_local_path=VEC,
            apply_spec={"phases": ["prompt"]},
        )


class TestFingerprintAndLengthSensitivity:
    def test_plain_spec_not_length_sensitive(self):
        req = to_engine_request(make_spec(phases=["prompt"]))
        assert not is_prompt_length_sensitive(req)

    @pytest.mark.parametrize(
        "apply_kwargs",
        [
            {"phases": ["prompt"], "positions": [-1]},
            {"phases": ["generation"], "generation_window": (0, 2)},
        ],
    )
    def test_length_sensitive_specs(self, apply_kwargs):
        assert is_prompt_length_sensitive(to_engine_request(make_spec(**apply_kwargs)))

    def test_fingerprint_stability_and_keying(self):
        fp_a1 = config_fingerprint(to_engine_request(make_spec(phases=["prompt"])))
        fp_a2 = config_fingerprint(to_engine_request(make_spec(phases=["prompt"])))
        fp_b = config_fingerprint(
            to_engine_request(make_spec(phases=["prompt"], exclude_positions=[0]))
        )
        assert fp_a1 == fp_a2
        assert fp_a1 != fp_b


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


class TestCollectorSemantics:
    """CPU-tensor checks of the v2 position collector.

    Scenarios simulate one request: a prefill step over prompt tokens,
    or a decode step processing generated token j (num_output = j + 1).
    """

    def test_prompt_phase_covers_all_prompt_tokens(self):
        assert run_collector(
            {"phases": ["prompt"]}, [11, 12, 13, 14], 0, False, 0, 4
        ) == [0, 1, 2, 3]

    def test_exclusions_compose_with_phase_wide_prompt(self):
        assert run_collector(
            {
                "phases": ["prompt"],
                "exclude_positions": [0],
                "exclude_tokens": [13],
            },
            [11, 12, 13, 14],
            0,
            False,
            0,
            4,
        ) == [1, 3]

    def test_generation_phase_skips_prefill_step(self):
        assert run_collector(
            {"phases": ["generation"]}, [11, 12, 13, 14], 0, False, 0, 4
        ) == []

    def test_window_exact_first_two_decode_steps(self):
        results = [
            run_collector(
                {"phases": ["generation"], "generation_window": (0, 2)},
                [99], 4 + j, True, j + 1, 4,
            )
            for j in range(4)
        ]
        assert results == [[0], [0], [], []]

    def test_window_skips_only_first_decode_step(self):
        results = [
            run_collector(
                {"phases": ["generation"], "generation_window": (1, None)},
                [99], 4 + j, True, j + 1, 4,
            )
            for j in range(3)
        ]
        assert results == [[], [0], [0]]

    def test_negative_position_resolves_against_prompt_length(self):
        assert run_collector(
            {"phases": ["prompt"], "positions": [-1]}, [11, 12], 2, False, 0, 4
        ) == [1]

    def test_token_and_position_triggers_union(self):
        assert run_collector(
            {"phases": ["prompt"], "tokens": [11], "positions": [3]},
            [11, 12, 13, 14],
            0,
            False,
            0,
            4,
        ) == [0, 3]


class TestTriggerControllerIntegration:
    def test_global_fast_path(self):
        ctrl = TriggerController()
        ctrl.configure_from_dict({"apply_spec": APPLY_ALL.to_wire()})
        assert ctrl.is_global_only_config()

    def test_filtered_spec_not_global(self):
        ctrl = TriggerController()
        ctrl.configure_from_dict(
            {"apply_spec": ApplySpec(phases=["prompt"], exclude_tokens=[3]).to_wire()}
        )
        assert not ctrl.is_global_only_config()
        assert ctrl.has_any_triggers()


class TestSelectSpec:
    """SelectSpec is the shared selection language; ApplySpec is its
    steering-facing name and capture consumes the same wire form."""

    def test_apply_spec_is_a_select_spec(self):
        from vllm.steer_vectors.api import SelectSpec

        assert issubclass(ApplySpec, SelectSpec)

    def test_wire_roundtrip(self):
        from vllm.steer_vectors.api import SelectSpec

        spec = SelectSpec(
            phases=["generation"],
            tokens=[5, 7],
            exclude_positions=[-1],
            generation_window=(0, 4),
        )
        rebuilt = SelectSpec.from_wire(spec.to_wire())
        assert rebuilt.to_wire() == spec.to_wire()

    def test_from_wire_rejects_unknown_fields(self):
        from vllm.steer_vectors.api import SelectSpec

        with pytest.raises(ValueError, match="unknown selection fields"):
            SelectSpec.from_wire({"phases": ["prompt"], "tokns": [1]})

    def test_from_wire_validates_clause(self):
        from vllm.steer_vectors.api import SelectSpec

        with pytest.raises(ValidationError):
            SelectSpec.from_wire({"phases": [], "tokens": None})


class TestCaptureStreamConfigSelection:
    """StreamConfig translates legacy kwargs into a SelectSpec clause
    and validates select= at enable time."""

    def test_legacy_kwargs_translate_to_select(self):
        from vllm.capture.session import StreamConfig

        config = StreamConfig(positions=[0, -1], token_ids=[42])
        assert config.selects_rows
        assert config.select["phases"] == ["prompt", "generation"]
        assert config.select["tokens"] == [42]
        assert config.select["positions"] == [0, -1]

    def test_select_clause_validated(self):
        from vllm.capture.session import StreamConfig

        with pytest.raises(ValueError, match="unknown selection fields"):
            StreamConfig(select={"phase": ["prompt"]})

    def test_select_conflicts_with_legacy_kwargs(self):
        from vllm.capture.session import StreamConfig

        with pytest.raises(ValueError, match="legacy"):
            StreamConfig(
                select={"phases": ["prompt"]}, token_ids=[1]
            )

    def test_select_conflicts_with_reductions(self):
        from vllm.capture.session import StreamConfig

        with pytest.raises(ValueError, match="last"):
            StreamConfig(select={"phases": ["prompt"]}, positions="last")
