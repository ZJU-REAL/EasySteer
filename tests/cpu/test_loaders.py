# SPDX-License-Identifier: Apache-2.0
"""Algorithm load_from_path implementations against synthetic files.

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
    LinearTransformAlgorithm,
    LMSteerAlgorithm,
    LoReFTAlgorithm,
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


class TestDirectPt:
    @pytest.fixture()
    def pt_path(self, tmp_path):
        path = os.path.join(tmp_path, "vec.pt")
        torch.save(torch.arange(8, dtype=torch.float32), path)
        return path

    def test_single_layer_pt(self, pt_path):
        out = DirectAlgorithm.load_from_path(
            pt_path, "cpu", config=CFG, target_layers=[7]
        )
        assert set(out["layer_payloads"]) == {7}

    def test_pt_without_target_layers_rejected(self, pt_path):
        with pytest.raises(ValueError):
            DirectAlgorithm.load_from_path(pt_path, "cpu", config=CFG)


class TestLinear:
    @pytest.fixture()
    def pkl_path(self, tmp_path):
        path = os.path.join(tmp_path, "linear.pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "A_": np.eye(4, dtype=np.float32),
                    "B_": np.zeros(4, dtype=np.float32),
                },
                f,
            )
        return path

    def test_linear_payloads(self, pkl_path):
        out = LinearTransformAlgorithm.load_from_path(
            pkl_path, "cpu", config=CFG, target_layers=[1, 2]
        )
        assert set(out["layer_payloads"]) == {1, 2}
        assert out["layer_payloads"][1]["weight"].shape == (4, 4)

    def test_linear_without_target_layers_rejected(self, pkl_path):
        with pytest.raises(ValueError):
            LinearTransformAlgorithm.load_from_path(pkl_path, "cpu", config=CFG)

    def test_linear_missing_matrix_rejected(self, tmp_path):
        bad_pkl = os.path.join(tmp_path, "bad.pkl")
        with open(bad_pkl, "wb") as f:
            pickle.dump({"C_": 1}, f)
        with pytest.raises(ValueError):
            LinearTransformAlgorithm.load_from_path(
                bad_pkl, "cpu", config=CFG, target_layers=[0]
            )


class TestLMSteer:
    def test_dict_checkpoint(self, tmp_path):
        path = os.path.join(tmp_path, "lms.pt")
        torch.save(
            {"projector1": torch.ones(8, 2), "projector2": torch.ones(8, 2)}, path
        )
        out = LMSteerAlgorithm.load_from_path(
            path, "cpu", config=CFG, target_layers=[3]
        )
        assert set(out["layer_payloads"]) == {3}
        with pytest.raises(ValueError):
            LMSteerAlgorithm.load_from_path(path, "cpu", config=CFG)

    def test_gpt2_style_list_checkpoint(self, tmp_path):
        path = os.path.join(tmp_path, "lms_list.pt")
        torch.save(
            [None, {"projector1": torch.ones(8, 2), "projector2": torch.ones(8, 2)}],
            path,
        )
        out = LMSteerAlgorithm.load_from_path(
            path, "cpu", config=CFG, target_layers=[0]
        )
        assert set(out["layer_payloads"]) == {0}


class TestReft:
    def test_direct_reft_dir(self, tmp_path):
        reft_dir = os.path.join(tmp_path, "reft")
        os.makedirs(reft_dir)
        with open(os.path.join(reft_dir, "reft_config.json"), "w") as f:
            json.dump({"representations": [{"layer": 3}]}, f)
        torch.save({"bias": torch.ones(8)}, os.path.join(reft_dir, "intervention.bin"))
        out = DirectAlgorithm.load_from_path(reft_dir, "cpu", config=CFG)
        assert set(out["layer_payloads"]) == {3}
        with pytest.raises(ValueError):
            DirectAlgorithm.load_from_path(
                reft_dir, "cpu", config=CFG, target_layers=[9]
            )

    def test_loreft_dir(self, tmp_path):
        loreft_dir = os.path.join(tmp_path, "loreft")
        os.makedirs(loreft_dir)
        with open(os.path.join(loreft_dir, "reft_config.json"), "w") as f:
            json.dump({"representations": [{"layer": 2}]}, f)
        torch.save(
            {
                "rotate_layer.parametrizations.weight.original": torch.ones(8, 4),
                "learned_source.weight": torch.ones(4, 8),
                "learned_source.bias": torch.ones(4),
            },
            os.path.join(loreft_dir, "intervention.bin"),
        )
        out = LoReFTAlgorithm.load_from_path(loreft_dir, "cpu", config=CFG)
        payload = out["layer_payloads"][2]
        assert payload["rotate_layer"] is not None
        assert payload["learned_source_weight"] is not None


def test_concept_replace_dir(tmp_path):
    cr_dir = os.path.join(tmp_path, "concept")
    os.makedirs(cr_dir)
    write_gguf(os.path.join(cr_dir, "h1.gguf"), {0: np.ones(8, dtype=np.float32)})
    write_gguf(os.path.join(cr_dir, "h2.gguf"), {0: 3 * np.ones(8, dtype=np.float32)})
    out = ConceptReplaceAlgorithm.load_from_path(cr_dir, "cpu", config=CFG)
    assert set(out["layer_payloads"]) == {0}
    assert float(out["layer_payloads"][0]["h2"][0]) == 3.0


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
