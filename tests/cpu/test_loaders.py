# SPDX-License-Identifier: Apache-2.0
"""Engine GGUF loading and client payload adapters against synthetic files.

Covers admission snapshots, explicit client adapters and malformed inputs.
Workers receive only the resulting canonical content, never source paths.
"""

import json
import os
import pickle

import numpy as np
import pytest
import torch
from vllm.model_hooks.steering.loading import resolve_vector_payload
from vllm.model_hooks.steering.payloads import materialize, validate_router_mode


def load_source(path, algorithm, **params):
    wire = resolve_vector_payload(path, None, algorithm, params=params)
    return materialize(wire, "cpu", torch.float32, None)


def write_gguf(path, layers):
    import gguf

    writer = gguf.GGUFWriter(path, "steervector")
    for layer, vec in layers.items():
        writer.add_tensor(f"direction.{layer}", np.asarray(vec, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture()
def gguf_path(tmp_path):
    path = os.path.join(tmp_path, "vec.gguf")
    write_gguf(
        path,
        {0: np.ones(8, dtype=np.float32), 5: 2 * np.ones(8, dtype=np.float32)},
    )
    return path


class TestGgufReaders:
    def test_loader_uses_explicit_algorithm(self, gguf_path):
        assert set(load_source(gguf_path, "direct")) == {0, 5}
        with pytest.raises(ValueError, match="plain path"):
            resolve_vector_payload(gguf_path + "|direct", None, "direct")

    @pytest.mark.parametrize("algorithm", ["direct", "erase", "replace"])
    def test_shared_gguf_reader(self, algorithm, gguf_path):
        payloads = load_source(gguf_path, algorithm)
        assert set(payloads) == {0, 5}
        assert payloads[5].dtype == torch.float32
        assert float(payloads[5][0]) == 2.0

    def test_erase_rejects_non_gguf(self, tmp_path):
        with pytest.raises(ValueError, match="only loads .gguf"):
            load_source(os.path.join(tmp_path, "x.pt"), "erase")

    def test_cached_snapshot_survives_rewrite_and_detects_new_version(
        self, gguf_path, monkeypatch
    ):
        """Stat the real source version; parse/hash unchanged content only once."""
        from vllm.model_hooks.steering import loading

        original_read = loading._read_payload
        reads = []

        def read(*args):
            reads.append(args)
            return original_read(*args)

        monkeypatch.setattr(loading, "_read_payload", read)
        first = resolve_vector_payload(gguf_path, None, "direct")
        same = resolve_vector_payload(gguf_path, None, "erase")
        assert same == first and len(reads) == 1
        first["tensors"]["layer.5"]["shape"][0] = 100
        assert resolve_vector_payload(gguf_path, None, "direct") == same

        before = os.stat(gguf_path)
        write_gguf(gguf_path, {0: np.zeros(8), 5: np.ones(8)})
        os.utime(gguf_path, ns=(before.st_atime_ns, before.st_mtime_ns))
        changed = resolve_vector_payload(gguf_path, None, "direct")
        assert changed["sha256"] != same["sha256"] and len(reads) == 2
        assert materialize(same, "cpu", torch.float32, None)[5][0] == 2

    def test_concept_pair_tracks_children_and_rejects_ambiguous_roles(self, tmp_path):
        h1, h2 = str(tmp_path / "h1.gguf"), str(tmp_path / "h2.gguf")
        write_gguf(h1, {3: np.ones(8)})
        write_gguf(h2, {3: np.zeros(8)})
        first = resolve_vector_payload(str(tmp_path), None, "concept_replace")
        before = os.stat(tmp_path)
        write_gguf(h2, {3: np.full(8, 2.0)})
        os.utime(tmp_path, ns=(before.st_atime_ns, before.st_mtime_ns))
        changed = resolve_vector_payload(str(tmp_path), None, "concept_replace")
        assert changed["sha256"] != first["sha256"]
        assert materialize(first, "cpu", torch.float32, None)[3]["h2"].sum() == 0
        write_gguf(str(tmp_path / "other_h1.gguf"), {3: np.ones(8)})
        with pytest.raises(ValueError, match="exactly one named h1"):
            resolve_vector_payload(str(tmp_path), None, "concept_replace")

    def test_changed_during_read_is_retried_as_one_snapshot(
        self, gguf_path, monkeypatch
    ):
        from vllm.model_hooks.steering import loading

        original_read = loading._read_payload
        reads = []

        def rewrite_after_read(*args):
            payload = original_read(*args)
            reads.append(payload)
            if len(reads) == 1:
                write_gguf(gguf_path, {0: np.full(8, 3.0)})
            return payload

        monkeypatch.setattr(loading, "_read_payload", rewrite_after_read)
        wire = resolve_vector_payload(gguf_path, None, "direct")
        assert len(reads) == 2
        assert materialize(wire, "cpu", torch.float32, None)[0][0] == 3


class TestPayloadAdapters:
    """Client-side adapters replace the deleted engine file heuristics."""

    def test_pt_direction(self, tmp_path):
        from vllm.model_hooks.steering.payloads import materialize

        import easysteer.vectors as vec

        path = os.path.join(tmp_path, "vec.pt")
        torch.save(torch.arange(8, dtype=torch.float32), path)
        payload = vec.from_pt_direction(path, layers=[7])
        out = materialize(payload.to_wire(), "cpu", torch.float32, None)
        assert set(out) == {7}
        with pytest.raises(ValueError, match="layers"):
            vec.from_pt_direction(path, layers=[])

    def test_explicit_format_entry_point(self, tmp_path, gguf_path):
        import easysteer.vectors as vec

        path = str(tmp_path / "direction.pt")
        torch.save(torch.arange(8, dtype=torch.float32), path)
        payload = vec.load(path, format="pt_direction", layers=[7])
        assert payload.to_wire() == vec.from_pt_direction(path, layers=[7]).to_wire()
        assert vec.load(gguf_path, format="gguf").to_wire() == (
            resolve_vector_payload(gguf_path, None, "direct")
        )
        with pytest.raises(ValueError, match="Unknown checkpoint format"):
            vec.load(path, format="pt")

    def test_linear_transport(self, tmp_path):
        from vllm.model_hooks.steering.payloads import materialize

        import easysteer.vectors as vec

        path = os.path.join(tmp_path, "linear.pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "A_": np.eye(4, dtype=np.float32),
                    "B_": np.zeros(4, dtype=np.float32),
                },
                f,
            )
        payload = vec.from_linear_transport(path)
        out = materialize(payload.to_wire(), "cpu", torch.float32, [1, 2])
        assert set(out) == {1, 2}
        assert out[1]["weight"].shape == (4, 4)

        bad = os.path.join(tmp_path, "bad.pkl")
        with open(bad, "wb") as f:
            pickle.dump({"C_": 1}, f)
        with pytest.raises(ValueError, match="A_"):
            vec.from_linear_transport(bad)

    def test_lm_steer_checkpoints(self, tmp_path):
        from vllm.model_hooks.steering.payloads import materialize

        import easysteer.vectors as vec

        path = os.path.join(tmp_path, "lms.pt")
        torch.save(
            {"projector1": torch.ones(8, 2), "projector2": torch.ones(8, 2)}, path
        )
        out = materialize(vec.from_lm_steer(path).to_wire(), "cpu", torch.float32, [3])
        assert set(out) == {3}

        gpt2_style = os.path.join(tmp_path, "lms_list.pt")
        torch.save(
            [None, {"projector1": torch.ones(8, 2), "projector2": torch.ones(8, 2)}],
            gpt2_style,
        )
        out = materialize(
            vec.from_lm_steer(gpt2_style).to_wire(), "cpu", torch.float32, [0]
        )
        assert set(out) == {0}

    def test_lm_steer_multivector_index_is_explicit(self, tmp_path):
        import easysteer.vectors as vec

        path = os.path.join(tmp_path, "stack.pt")
        torch.save(
            {
                "projector1": torch.ones(2, 8, 2),
                "projector2": torch.ones(2, 8, 2),
            },
            path,
        )
        payload = vec.from_lm_steer(path, vector_index=1)
        assert payload.projector1.shape == (8, 2)
        with pytest.raises(ValueError, match="out of range"):
            vec.from_lm_steer(path, vector_index=5)


class TestReft:
    @staticmethod
    def config(layer, intervention="loreft"):
        name = "BiasIntervention" if intervention == "bias" else "LoreftIntervention"
        return {
            "representations": [
                {
                    "layer": layer,
                    "component": "block_output",
                    "unit": "pos",
                    "max_number_of_units": 1,
                }
            ],
            "intervention_types": [
                f"<class 'easysteer.reft.pyreft.reft.algorithms.{intervention}.{name}'>"
            ],
        }

    def test_bias_intervention_dir(self, tmp_path):
        from vllm.model_hooks.steering.payloads import DirectionVector, materialize

        import easysteer.vectors as vec

        reft_dir = os.path.join(tmp_path, "reft")
        os.makedirs(reft_dir)
        with open(os.path.join(reft_dir, "reft_config.json"), "w") as f:
            json.dump(self.config(3, "bias"), f)
        torch.save({"bias": torch.ones(8)}, os.path.join(reft_dir, "intervention.bin"))
        payload = vec.from_pyreft(reft_dir)
        assert isinstance(payload, DirectionVector)
        out = materialize(payload.to_wire(), "cpu", torch.float32, None)
        assert set(out) == {3}

    def test_loreft_dir(self, tmp_path):
        from vllm.model_hooks.steering.payloads import ReftIntervention, materialize

        import easysteer.vectors as vec

        loreft_dir = os.path.join(tmp_path, "loreft")
        os.makedirs(loreft_dir)
        with open(os.path.join(loreft_dir, "reft_config.json"), "w") as f:
            json.dump(self.config(2), f)
        torch.save(
            {
                "rotate_layer": torch.ones(8, 2),
                "learned_source.weight": torch.ones(2, 8),
                "learned_source.bias": torch.ones(2),
            },
            os.path.join(loreft_dir, "intervention.bin"),
        )
        payload = vec.from_pyreft(loreft_dir)
        assert isinstance(payload, ReftIntervention)
        assert payload.layer == 2
        out = materialize(payload.to_wire(), "cpu", torch.float32, None)
        assert set(out) == {2}
        assert out[2]["rotate_layer"].shape == (8, 2)

    def test_loreft_dir_bare_keys(self, tmp_path):
        # pyreft's save() also emits LoReFT state dicts with unprefixed
        # weight/bias next to rotate_layer.
        from vllm.model_hooks.steering.payloads import ReftIntervention

        import easysteer.vectors as vec

        loreft_dir = os.path.join(tmp_path, "loreft_bare")
        os.makedirs(loreft_dir)
        with open(os.path.join(loreft_dir, "reft_config.json"), "w") as f:
            config = self.config(8)
            config["intervention_types"] = [
                "<class 'pyreft.interventions.LoreftIntervention'>"
            ]
            json.dump(config, f)
        torch.save(
            {
                "weight": torch.ones(2, 8),
                "bias": torch.ones(2),
                "rotate_layer": torch.ones(8, 2),
            },
            os.path.join(loreft_dir, "intervention.bin"),
        )
        payload = vec.from_pyreft(loreft_dir)
        assert isinstance(payload, ReftIntervention)
        assert payload.layer == 8
        assert payload.learned_source_weight.shape == (2, 8)
        assert payload.learned_source_bias.shape == (2,)

    @pytest.mark.parametrize(
        "invalid",
        ["component", "unit", "count", "type", "subspace", "act_fn", "training_act_fn"],
    )
    def test_unsupported_reft_semantics_rejected_before_tensor_loading(
        self, tmp_path, monkeypatch, invalid
    ):
        import easysteer.vectors as vec

        config = self.config(2)
        representation = config["representations"][0]
        if invalid == "component":
            # This component can have hidden_size width, so shape checks cannot catch it.
            representation["component"] = "attention_output"
        elif invalid == "unit":
            representation["unit"] = "h.pos"
        elif invalid == "count":
            config["representations"].append(dict(representation))
        elif invalid == "type":
            config["intervention_types"] = [
                "<class 'pyreft.interventions.ConsreftIntervention'>"
            ]
        elif invalid == "subspace":
            representation["subspace_partition"] = [[0, 2]]
        elif invalid == "act_fn":
            config["act_fn"] = "relu"
        else:
            config["easysteer_training"] = {"act_fn": "relu"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "intervention.bin").touch()
        monkeypatch.setattr(
            torch,
            "load",
            lambda *args, **kwargs: pytest.fail(
                "weights read before validating target"
            ),
        )
        with pytest.raises(ValueError, match="Unsupported|exactly one"):
            vec.from_pyreft(str(tmp_path))


class TestMoeRouterJson:
    @staticmethod
    def write_moe(tmp_path, name, layer_configs):
        path = os.path.join(tmp_path, name)
        with open(path, "w") as f:
            json.dump({"layer_configs": layer_configs}, f)
        return path

    def test_valid_config_with_canonical_modes(self, tmp_path):
        path = self.write_moe(
            tmp_path,
            "moe.json",
            {
                "1": {"expert_ids": [1, 2], "mode": "deactivate"},
                "2": {"expert_ids": [3], "mode": "activate"},
                "3": {"expert_ids": [4], "mode": "soft", "lambda": 0.7},
            },
        )
        payloads = load_source(path, "moe_router")
        assert set(payloads) == {1, 2, 3}
        assert payloads[2]["mode"] == "activate"
        assert payloads[3]["lambda"] == 0.7

    def test_file_and_inline_have_same_content_identity(self, tmp_path):
        config = {"2": {"mode": "soft_topk", "expert_ids": [3], "lambda": 0.7}}
        path = self.write_moe(tmp_path, "topk.json", config)
        file_wire = resolve_vector_payload(path, None, "moe_router", params={"topk": 4})
        inline = resolve_vector_payload(
            None,
            None,
            "moe_router",
            [2],
            {"mode": "soft_topk", "expert_ids": [3], "lambda": 0.7, "topk": 4},
        )
        assert file_wire == inline
        override = resolve_vector_payload(
            path, None, "moe_router", params={"mode": "deactivate"}
        )
        assert override["extra"]["layers"]["2"]["mode"] == "deactivate"
        assert override["sha256"] != file_wire["sha256"]

    def test_router_layer_selection_does_not_change_preload_identity(self, tmp_path):
        path = self.write_moe(
            tmp_path,
            "subset.json",
            {
                "1": {"expert_ids": [1]},
                "2": {"expert_ids": [2], "mode": "soft_random"},
            },
        )
        all_layers = resolve_vector_payload(path, None, "moe_router")
        subset = resolve_vector_payload(path, None, "moe_router", [1])
        assert subset == all_layers

    @pytest.mark.parametrize(
        "layer_values",
        [
            {"mode": "soft_topk", "expert_ids": [2]},
            {"mode": "soft", "expert_ids": [2], "lambda": 0.7, "topk": 6},
        ],
    )
    def test_explicit_router_parameters_override_source_and_data_equally(
        self, tmp_path, layer_values
    ):
        import easysteer.vectors as vec

        path = self.write_moe(tmp_path, "overrides.json", {"1": layer_values})
        data = vec.load(path, format="moe_router")
        overrides = {"mode": "soft_topk", "lambda": 1.2, "topk": 2}
        source_wire = resolve_vector_payload(path, None, "moe_router", params=overrides)
        data_wire = resolve_vector_payload(None, data, "moe_router", params=overrides)
        adapted = vec.load(path, format="moe_router", **overrides)
        assert source_wire == data_wire == adapted.to_wire()
        config = source_wire["extra"]["layers"]["1"]
        assert config["mode"] == "soft_topk" and config["lambda"] == 1.2
        assert config["topk"] == 2 and config["expert_ids"] == [2]

    def test_unspecified_router_parameters_preserve_existing_values(self, tmp_path):
        path = self.write_moe(
            tmp_path,
            "preserved.json",
            {"1": {"mode": "soft_topk", "expert_ids": [2], "lambda": 0.7, "topk": 6}},
        )
        wire = resolve_vector_payload(path, None, "moe_router", params={"lambda": 1.2})
        assert wire["extra"]["layers"]["1"] == {
            "mode": "soft_topk",
            "expert_ids": [2],
            "lambda": 1.2,
            "topk": 6,
        }

    def test_router_mode_switch_starts_from_same_canonical_source_and_data(
        self, tmp_path
    ):
        import easysteer.vectors as vec

        path = self.write_moe(
            tmp_path,
            "switch-mode.json",
            {"1": {"mode": "activate", "expert_ids": [2], "lambda": 4.0}},
        )
        data = vec.load(path, format="moe_router")
        overrides = {"mode": "soft"}
        source_wire = resolve_vector_payload(path, None, "moe_router", params=overrides)
        data_wire = resolve_vector_payload(None, data, "moe_router", params=overrides)
        assert source_wire == data_wire
        assert source_wire["extra"]["layers"]["1"]["lambda"] == 0.5

    def test_expert_ids_only_configure_pure_inline_router_input(self, tmp_path):
        from vllm.model_hooks.steering.api import ApplySpec, VectorSpec
        from vllm.model_hooks.steering.payloads import RouterConfig

        path = self.write_moe(tmp_path, "experts.json", {"1": {"expert_ids": [2]}})
        data = RouterConfig({1: {"expert_ids": [2]}})
        for source, payload in ((path, None), (None, data)):
            with pytest.raises(ValueError, match="unknown params.*expert_ids"):
                VectorSpec(
                    source=source,
                    data=payload,
                    algorithm="moe_router",
                    params={"expert_ids": [3]},
                    apply=ApplySpec(generation="all"),
                )
            with pytest.raises(ValueError, match="unknown params.*expert_ids"):
                resolve_vector_payload(
                    source, payload, "moe_router", params={"expert_ids": [3]}
                )
        inline = resolve_vector_payload(
            None, None, "moe_router", [1], {"expert_ids": [3]}
        )
        assert inline["extra"]["layers"]["1"]["expert_ids"] == [3]

    @pytest.mark.parametrize("mode", ["boost", "suppress", "soft_hard", "steermoe"])
    def test_removed_mode_aliases_rejected(self, tmp_path, mode):
        with pytest.raises(ValueError, match="unknown moe_router mode"):
            validate_router_mode(mode)
        path = self.write_moe(
            tmp_path, "old-mode.json", {"1": {"expert_ids": [1], "mode": mode}}
        )
        with pytest.raises(ValueError, match="Layer 1: unknown moe_router mode"):
            load_source(path, "moe_router")

    @pytest.mark.parametrize(
        "name, layer_configs",
        [
            ("moe_bad_layer.json", {"abc": {"expert_ids": [1]}}),
            ("moe_bad_mode.json", {"1": {"expert_ids": [1], "mode": "x"}}),
            ("moe_no_ids.json", {"1": {"mode": "deactivate"}}),
        ],
    )
    def test_invalid_configs_rejected(self, tmp_path, name, layer_configs):
        with pytest.raises(ValueError):
            load_source(self.write_moe(tmp_path, name, layer_configs), "moe_router")


@pytest.mark.parametrize("params", [[], "mode=activate", 0])
def test_shared_resolver_rejects_non_dict_parameters(params):
    with pytest.raises(ValueError, match="parameters must be a dictionary"):
        resolve_vector_payload(None, None, "moe_router", [1], params)


@pytest.mark.parametrize(
    "algorithm, params",
    [
        ("direct", {"mode": "activate"}),
        ("moe_router", {"expert_ids": [1], "top_k": 2}),
    ],
)
def test_shared_resolver_rejects_unknown_parameters_before_loading(algorithm, params):
    with pytest.raises(ValueError, match="unknown params for algorithm"):
        resolve_vector_payload("missing.gguf", None, algorithm, params=params)
