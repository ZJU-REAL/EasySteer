"""The JSON producer preserves the engine's canonical tensor bytes and hash."""

import base64
import json
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("kind", ["direction", "linear", "lowrank", "reft", "concept_pair", "router"])
def test_json_payload_round_trip(payload_modules, kind):
    payloads, vectors = payload_modules
    payload = {
        "direction": lambda: payloads.DirectionVector({8: [1.0, -2.0]}),
        "linear": lambda: payloads.LinearMap([[1, 0], [0, 1]], [0.5, -0.5]),
        "lowrank": lambda: payloads.LowRankProjector([[1], [2]], [[3], [4]]),
        "reft": lambda: payloads.ReftIntervention([[1], [0]], [[0, 1]], [0.5], layer=22),
        "concept_pair": lambda: payloads.ConceptPair({8: [1, 2]}, {8: [3, 4]}),
        "router": lambda: payloads.RouterConfig({8: {"mode": "soft", "expert_ids": [1], "lambda": 0.7}}),
    }[kind]()
    raw = payload.to_wire()
    decoded = json.loads(json.dumps(vectors.to_json_payload(payload)))
    assert payloads.validate_wire(decoded) == kind
    assert decoded["sha256"] == raw["sha256"]
    for name, tensor in decoded["tensors"].items():
        tensor["data"] = base64.b64decode(tensor["data"], validate=True)
    assert decoded == raw
    assert payload.to_wire() == raw


def test_space_light_runtime_accepts_router_metadata(payload_modules, tmp_path):
    payloads, vectors = payload_modules
    payload = payloads.RouterConfig({8: {"mode": "deactivate", "expert_ids": [1]}})
    path = tmp_path / "router-payload.json"
    wire = vectors.to_json_payload(payload)
    path.write_text(json.dumps(wire))
    runtime_path = Path(__file__).resolve().parents[2] / "hf-space/runtime.py"
    spec = importlib.util.spec_from_file_location("space_router_runtime", runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    loaded = runtime.load_payload(path, "moe_router")
    assert payloads.validate_wire(loaded) == "router"
    assert loaded == wire


@pytest.mark.parametrize("algorithm,adapter", [
    ("linear", "from_linear_transport"),
    ("lm_steer", "from_lm_steer"),
    ("loreft", "from_pyreft"),
])
def test_space_export_is_readable_by_light_runtime(payload_modules, monkeypatch, tmp_path, algorithm, adapter):
    payloads, vectors = payload_modules
    payload = {
        "linear": lambda: payloads.LinearMap([[1]]),
        "lm_steer": lambda: payloads.LowRankProjector([[1]], [[1]]),
        "loreft": lambda: payloads.ReftIntervention([[1]], [[1]], layer=22),
    }[algorithm]()
    monkeypatch.setattr(vectors, adapter, lambda path: payload)
    space = Path(__file__).resolve().parents[2] / "hf-space"

    def load(name):
        spec = importlib.util.spec_from_file_location(f"test_space_{name}", space / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"mock checkpoint")
    output = tmp_path / "payload.json"
    load("export_payload").export_payload(checkpoint, algorithm, output)
    wire = load("runtime").load_payload(output, algorithm)
    assert payloads.validate_wire(wire) == payload.kind
    assert wire == vectors.to_json_payload(payload)
