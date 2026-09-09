"""Check pure preset conversion against the current engine contract."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SPACE = Path(__file__).resolve().parents[1]
ENGINE = SPACE.parent / "vllm-steer/vllm"
spec = importlib.util.spec_from_file_location("space_steering_config", SPACE / "steering_config.py")
config_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_builder)


@pytest.fixture
def steering_spec(monkeypatch):
    for name, path in (
        ("vllm", ENGINE),
        ("vllm.model_hooks", ENGINE / "model_hooks"),
        ("vllm.model_hooks.selection", ENGINE / "model_hooks/selection"),
        ("vllm.model_hooks.steering", ENGINE / "model_hooks/steering"),
    ):
        module = ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)
    for name in (
        "selection.schema",
        "selection.spec",
        "steering.capabilities",
        "steering.payloads",
        "steering.input_validation",
        "steering.api",
    ):
        full_name = f"vllm.model_hooks.{name}"
        path = ENGINE / "model_hooks" / f"{name.replace('.', '/')}.py"
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
    return sys.modules["vllm.model_hooks.steering.api"].SteeringSpec


@pytest.mark.parametrize("fields,expected", [
    ({}, {"prompt": "all", "generation": "all"}),
    ({"prefill_trigger_positions": "-1"}, {"prompt_positions": [-1]}),
    ({"prefill_trigger_tokens": "7, 8", "generate_trigger_tokens": "-1"},
     {"prompt_tokens": [7, 8], "generation": "all"}),
    ({"prefill_trigger_tokens": "-1", "prefill_trigger_positions": "-1", "generate_trigger_tokens": "9"},
     {"prompt": "all", "generation_tokens": [9]}),
    ({"prefill_trigger_positions": 0, "generate_trigger_tokens": "0"},
     {"prompt_positions": [0], "generation_tokens": [0]}),
])
def test_phase_selection_preserves_trigger_meaning(steering_spec, fields, expected):
    config = {"steer_vector": {"path": "vector.gguf", "scale": "2.5", **fields}}
    wire = config_builder.build_single_spec_wire(config, lambda algorithm, path, payload: {"source": path}, scale_override=0)
    assert wire["vectors"][0]["scale"] == 0
    assert wire["vectors"][0]["apply"] == expected
    steering_spec.model_validate(wire)


@pytest.mark.parametrize("preset", sorted((SPACE / "configs").glob("*/*.json")), ids=lambda path: path.stem)
def test_every_bundled_preset_is_an_engine_spec(steering_spec, preset):
    config = json.loads(preset.read_text())

    def source(algorithm, path, payload_path):
        if payload_path:
            return {"data": json.loads((SPACE / payload_path).read_text())}
        return {"source": str(SPACE / path)}

    if "vector_configs" in config:
        wire = config_builder.build_multi_spec_wire(config, source)
        assert len(wire["vectors"]) == len(config["vector_configs"])
        assert [vector["apply"]["prompt_positions"] for vector in wire["vectors"]] == [[-1], [-2], [-3], [-4]]
        assert wire["conflict"] == "sequential"
    else:
        wire = config_builder.build_single_spec_wire(config, source)
    steering_spec.model_validate(wire)
