# SPDX-License-Identifier: Apache-2.0
"""Engine GGUF loading and client payload adapters against synthetic files.

Covers the shared GGUF/ReFT helpers, the unified keyword signature, and
the explicit failure paths (missing target_layers, bad MoE modes) that
used to be silent defaults or warn-and-skip.
"""

import json
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.steer_vectors.algorithms import (
    DirectAlgorithm,
    EraseAlgorithm,
    MoERouterAlgorithm,
    ReplaceAlgorithm,
)
from vllm.steer_vectors.algorithms.concept_replace import ConceptReplaceAlgorithm

CFG = SimpleNamespace(adapter_dtype=torch.float32)


def write_gguf(path, layers):
    import gguf

    writer = gguf.GGUFWriter(path, "steervector")
    for layer, vec in layers.items():
        writer.add_tensor(f"direction.{layer}", vec)
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
    @pytest.mark.parametrize(
        "algo_cls", [DirectAlgorithm, EraseAlgorithm, ReplaceAlgorithm]
    )
    def test_shared_gguf_reader(self, algo_cls, gguf_path):
        payloads = algo_cls.load_from_path(gguf_path, "cpu", config=CFG)[
            "layer_payloads"
        ]
        assert set(payloads) == {0, 5}
        assert payloads[5].dtype == torch.float32
        assert float(payloads[5][0]) == 2.0

    def test_erase_rejects_non_gguf(self, tmp_path):
        with pytest.raises(ValueError):
            EraseAlgorithm.load_from_path(
                os.path.join(tmp_path, "x.pt"), "cpu", config=CFG
            )


class TestPayloadAdapters:
    """Client-side adapters replace the deleted engine file heuristics."""

    def test_pt_direction(self, tmp_path):
        import easysteer.vectors as vec
        from vllm.steer_vectors.payloads import materialize

        path = os.path.join(tmp_path, "vec.pt")
        torch.save(torch.arange(8, dtype=torch.float32), path)
        payload = vec.from_pt_direction(path, layers=[7])
        out = materialize(payload.to_wire(), "cpu", torch.float32, None)
        assert set(out) == {7}
        with pytest.raises(ValueError, match="layers"):
            vec.from_pt_direction(path, layers=[])

    def test_linear_transport(self, tmp_path):
        import easysteer.vectors as vec
        from vllm.steer_vectors.payloads import materialize

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
        import easysteer.vectors as vec
        from vllm.steer_vectors.payloads import materialize

        path = os.path.join(tmp_path, "lms.pt")
        torch.save(
            {"projector1": torch.ones(8, 2), "projector2": torch.ones(8, 2)}, path
        )
        out = materialize(
            vec.from_lm_steer(path).to_wire(), "cpu", torch.float32, [3]
        )
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
    def test_bias_intervention_dir(self, tmp_path):
        import easysteer.vectors as vec
        from vllm.steer_vectors.payloads import DirectionVector, materialize

        reft_dir = os.path.join(tmp_path, "reft")
        os.makedirs(reft_dir)
        with open(os.path.join(reft_dir, "reft_config.json"), "w") as f:
            json.dump({"representations": [{"layer": 3}]}, f)
        torch.save({"bias": torch.ones(8)}, os.path.join(reft_dir, "intervention.bin"))
        payload = vec.from_pyreft(reft_dir)
        assert isinstance(payload, DirectionVector)
        out = materialize(payload.to_wire(), "cpu", torch.float32, None)
        assert set(out) == {3}

    def test_loreft_dir(self, tmp_path):
        import easysteer.vectors as vec
        from vllm.steer_vectors.payloads import ReftIntervention, materialize

        loreft_dir = os.path.join(tmp_path, "loreft")
        os.makedirs(loreft_dir)
        with open(os.path.join(loreft_dir, "reft_config.json"), "w") as f:
            json.dump({"representations": [{"layer": 2}]}, f)
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
        import easysteer.vectors as vec
        from vllm.steer_vectors.payloads import ReftIntervention

        loreft_dir = os.path.join(tmp_path, "loreft_bare")
        os.makedirs(loreft_dir)
        with open(os.path.join(loreft_dir, "reft_config.json"), "w") as f:
            json.dump({"representations": [{"layer": 8}]}, f)
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


class TestMoeRouterJson:
    @staticmethod
    def write_moe(tmp_path, name, layer_configs):
        path = os.path.join(tmp_path, name)
        with open(path, "w") as f:
            json.dump({"layer_configs": layer_configs}, f)
        return path

    def test_valid_config_with_aliases(self, tmp_path):
        path = self.write_moe(
            tmp_path,
            "moe.json",
            {
                "1": {"expert_ids": [1, 2], "mode": "deactivate"},
                "2": {"expert_ids": [3], "mode": "boost"},
                "3": {"expert_ids": [4], "mode": "soft", "lambda": 0.7},
            },
        )
        payloads = MoERouterAlgorithm.load_from_path(path, "cpu", config=CFG)[
            "layer_payloads"
        ]
        assert set(payloads) == {1, 2, 3}
        assert payloads[2]["mode"] == "boost"
        assert payloads[3]["lambda"] == 0.7

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
            MoERouterAlgorithm.load_from_path(
                self.write_moe(tmp_path, name, layer_configs), "cpu", config=CFG
            )


def test_template_exposes_triggers():
    algo = DirectAlgorithm()
    assert hasattr(algo, "triggers") and not hasattr(algo, "params")
