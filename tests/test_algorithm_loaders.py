#!/usr/bin/env python3
"""CPU unit test for algorithm load_from_path implementations.

Exercises every algorithm's loader against synthetic files: the shared
GGUF/ReFT helpers, the unified keyword signature, and the explicit
failure paths (missing target_layers, bad MoE modes) that used to be
silent defaults or warn-and-skip.
"""

import json
import os
import pickle
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
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
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"OK: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        FAILURES.append(name)


def expect_raises(name, exc_type, fn):
    try:
        fn()
    except exc_type:
        print(f"OK: {name}")
    except Exception as e:  # noqa: BLE001 - report wrong exception type
        print(f"FAIL: {name} raised {type(e).__name__}: {e}")
        FAILURES.append(name)
    else:
        print(f"FAIL: {name} did not raise")
        FAILURES.append(name)


def write_gguf(path, layers):
    import gguf

    writer = gguf.GGUFWriter(path, "steervector")
    for layer, vec in layers.items():
        writer.add_tensor(f"direction.{layer}", vec)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    tmp = tempfile.mkdtemp(prefix="steer_loader_test_")

    # --- GGUF path (direct / erase / replace share the reader) ---
    gguf_path = os.path.join(tmp, "vec.gguf")
    write_gguf(
        gguf_path,
        {0: np.ones(8, dtype=np.float32), 5: 2 * np.ones(8, dtype=np.float32)},
    )
    for algo_cls in (DirectAlgorithm, EraseAlgorithm, ReplaceAlgorithm):
        out = algo_cls.load_from_path(gguf_path, "cpu", config=CFG)
        payloads = out["layer_payloads"]
        check(
            f"{algo_cls.__name__} gguf layers",
            set(payloads) == {0, 5}
            and payloads[5].dtype == torch.float32
            and float(payloads[5][0]) == 2.0,
        )
    expect_raises(
        "erase rejects non-gguf",
        ValueError,
        lambda: EraseAlgorithm.load_from_path(
            os.path.join(tmp, "x.pt"), "cpu", config=CFG
        ),
    )

    # --- direct .pt (single layer, requires target_layers) ---
    pt_path = os.path.join(tmp, "vec.pt")
    torch.save(torch.arange(8, dtype=torch.float32), pt_path)
    out = DirectAlgorithm.load_from_path(
        pt_path, "cpu", config=CFG, target_layers=[7]
    )
    check("direct .pt layer", set(out["layer_payloads"]) == {7})
    expect_raises(
        "direct .pt without target_layers",
        ValueError,
        lambda: DirectAlgorithm.load_from_path(pt_path, "cpu", config=CFG),
    )

    # --- linear (pkl, requires target_layers; no 48-layer fallback) ---
    pkl_path = os.path.join(tmp, "linear.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {"A_": np.eye(4, dtype=np.float32), "B_": np.zeros(4, dtype=np.float32)},
            f,
        )
    out = LinearTransformAlgorithm.load_from_path(
        pkl_path, "cpu", config=CFG, target_layers=[1, 2]
    )
    check(
        "linear payloads",
        set(out["layer_payloads"]) == {1, 2}
        and out["layer_payloads"][1]["weight"].shape == (4, 4),
    )
    expect_raises(
        "linear without target_layers",
        ValueError,
        lambda: LinearTransformAlgorithm.load_from_path(pkl_path, "cpu", config=CFG),
    )
    bad_pkl = os.path.join(tmp, "bad.pkl")
    with open(bad_pkl, "wb") as f:
        pickle.dump({"C_": 1}, f)
    expect_raises(
        "linear missing A_",
        ValueError,
        lambda: LinearTransformAlgorithm.load_from_path(
            bad_pkl, "cpu", config=CFG, target_layers=[0]
        ),
    )

    # --- lm_steer (.pt dict and gpt2-style list; requires target_layers) ---
    lms_path = os.path.join(tmp, "lms.pt")
    torch.save(
        {"projector1": torch.ones(8, 2), "projector2": torch.ones(8, 2)}, lms_path
    )
    out = LMSteerAlgorithm.load_from_path(
        lms_path, "cpu", config=CFG, target_layers=[3]
    )
    check("lm_steer dict payloads", set(out["layer_payloads"]) == {3})
    lms_list_path = os.path.join(tmp, "lms_list.pt")
    torch.save(
        [None, {"projector1": torch.ones(8, 2), "projector2": torch.ones(8, 2)}],
        lms_list_path,
    )
    out = LMSteerAlgorithm.load_from_path(
        lms_list_path, "cpu", config=CFG, target_layers=[0]
    )
    check("lm_steer list payloads", set(out["layer_payloads"]) == {0})
    expect_raises(
        "lm_steer without target_layers",
        ValueError,
        lambda: LMSteerAlgorithm.load_from_path(lms_path, "cpu", config=CFG),
    )

    # --- ReFT directory (direct + loreft share discovery) ---
    reft_dir = os.path.join(tmp, "reft")
    os.makedirs(reft_dir)
    with open(os.path.join(reft_dir, "reft_config.json"), "w") as f:
        json.dump({"representations": [{"layer": 3}]}, f)
    torch.save(
        {"bias": torch.ones(8)}, os.path.join(reft_dir, "intervention.bin")
    )
    out = DirectAlgorithm.load_from_path(reft_dir, "cpu", config=CFG)
    check("direct reft layer", set(out["layer_payloads"]) == {3})
    expect_raises(
        "reft layer mismatch",
        ValueError,
        lambda: DirectAlgorithm.load_from_path(
            reft_dir, "cpu", config=CFG, target_layers=[9]
        ),
    )

    loreft_dir = os.path.join(tmp, "loreft")
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
    check(
        "loreft payload keys",
        payload["rotate_layer"] is not None
        and payload["learned_source_weight"] is not None,
    )

    # --- concept_replace (two gguf files in a directory) ---
    cr_dir = os.path.join(tmp, "concept")
    os.makedirs(cr_dir)
    write_gguf(os.path.join(cr_dir, "h1.gguf"), {0: np.ones(8, dtype=np.float32)})
    write_gguf(
        os.path.join(cr_dir, "h2.gguf"), {0: 3 * np.ones(8, dtype=np.float32)}
    )
    out = ConceptReplaceAlgorithm.load_from_path(cr_dir, "cpu", config=CFG)
    check(
        "concept_replace payloads",
        set(out["layer_payloads"]) == {0}
        and float(out["layer_payloads"][0]["h2"][0]) == 3.0,
    )

    # --- moe_router JSON: valid config, aliases, explicit failures ---
    moe_path = os.path.join(tmp, "moe.json")
    with open(moe_path, "w") as f:
        json.dump(
            {
                "layer_configs": {
                    "1": {"expert_ids": [1, 2], "mode": "deactivate"},
                    "2": {"expert_ids": [3], "mode": "boost"},
                    "3": {"expert_ids": [4], "mode": "soft", "lambda": 0.7},
                }
            },
            f,
        )
    out = MoERouterAlgorithm.load_from_path(moe_path, "cpu", config=CFG)
    payloads = out["layer_payloads"]
    check(
        "moe payloads",
        set(payloads) == {1, 2, 3}
        and payloads[2]["mode"] == "boost"
        and payloads[3]["lambda"] == 0.7,
    )

    def write_moe(name, layer_configs):
        p = os.path.join(tmp, name)
        with open(p, "w") as f:
            json.dump({"layer_configs": layer_configs}, f)
        return p

    expect_raises(
        "moe invalid layer id",
        ValueError,
        lambda: MoERouterAlgorithm.load_from_path(
            write_moe("moe_bad_layer.json", {"abc": {"expert_ids": [1]}}),
            "cpu",
            config=CFG,
        ),
    )
    expect_raises(
        "moe unknown mode",
        ValueError,
        lambda: MoERouterAlgorithm.load_from_path(
            write_moe("moe_bad_mode.json", {"1": {"expert_ids": [1], "mode": "x"}}),
            "cpu",
            config=CFG,
        ),
    )
    expect_raises(
        "moe missing expert ids",
        ValueError,
        lambda: MoERouterAlgorithm.load_from_path(
            write_moe("moe_no_ids.json", {"1": {"mode": "deactivate"}}),
            "cpu",
            config=CFG,
        ),
    )

    # --- template surface: triggers rename ---
    algo = DirectAlgorithm()
    check(
        "template exposes triggers",
        hasattr(algo, "triggers") and not hasattr(algo, "params"),
    )

    print(f"\nOVERALL: {'FAIL' if FAILURES else 'PASS'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
