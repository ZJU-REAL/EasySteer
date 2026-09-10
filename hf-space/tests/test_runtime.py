"""The API Space must work without torch or vLLM installed."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import httpx
import pytest

SPACE = Path(__file__).resolve().parents[1]
ENGINE = SPACE.parent / "vllm-steer/vllm"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_module("space_runtime", SPACE / "runtime.py")
exporter = load_module("space_exporter", SPACE / "export_payload.py")


@pytest.fixture
def steering_spec(monkeypatch):
    """Load the engine's actual public schema without its GPU dependencies."""
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


def test_default_api_mode_requires_configuration():
    with pytest.raises(ValueError, match="VLLM_API_URL, VLLM_MODEL_NAME"):
        runtime.demo_mode({})
    assert (
        runtime.demo_mode(
            {"VLLM_API_URL": "http://server/v1", "VLLM_MODEL_NAME": "model"}
        )
        == "api"
    )
    assert runtime.demo_mode({"DEMO_MODE": "gpu"}) == "gpu"
    for invalid in ("typo", "auto"):
        with pytest.raises(ValueError, match="DEMO_MODE"):
            runtime.demo_mode({"DEMO_MODE": invalid})


def test_export_reports_lfs_pointer_before_importing_engine(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\n"
    )
    with pytest.raises(ValueError, match="Git LFS pointer"):
        exporter.export_payload(checkpoint, "loreft", tmp_path / "payload.json")


def test_payload_kind_must_match_preset(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"version": 1, "kind": "direction"}))
    with pytest.raises(ValueError, match="not a version 1 loreft payload"):
        runtime.load_payload(path, "loreft")


def test_api_ui_and_bundled_payload(monkeypatch):
    monkeypatch.syspath_prepend(str(SPACE))
    monkeypatch.setenv("DEMO_MODE", "api")
    monkeypatch.setenv("VLLM_API_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("VLLM_MODEL_NAME", "qwen-demo")
    monkeypatch.setenv("VLLM_VECTOR_BASE_PATH", "/remote/hf-space")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    before = set(sys.modules)
    app = load_module("space_api_app", SPACE / "app.py")
    assert not {"torch", "vllm", "easysteer"} & (set(sys.modules) - before)
    direct = app.load_steering_spec(
        app.SINGLE_CONFIGS["emotion_direct"],
        app._resolve_path,
        app_dir=app.APP_DIR,
    )
    assert direct["vectors"][0]["source"].startswith("/remote/hf-space/")
    reft = app.load_steering_spec(
        app.SINGLE_CONFIGS["emoji_loreft"],
        app._resolve_path,
        app_dir=app.APP_DIR,
    )
    vector = reft["vectors"][0]
    assert vector["data"]["kind"] == "reft"
    assert vector["data"]["extra"]["layer"] == 22
    assert "source" not in vector
    assert vector["apply"] == {"prompt_positions": [-1]}
    requests = []

    def respond(request):
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen-demo",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "mock reply"},
                    }
                ],
            },
        )

    with app.OpenAI(
        base_url="http://127.0.0.1:1/v1",
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(app, "_api_client", client)
        assert app.generate_single(
            "emoji_loreft", "Who are you?", 1.0, progress=lambda *args, **kwargs: None
        ) == ("mock reply", "mock reply")
        assert len(requests) == 2
        assert requests[0]["steering"] is False
        assert requests[1]["steering"] == reft

        requests.clear()
        multi = app.load_steering_spec(
            app.MULTI_CONFIGS["refusal_direction"],
            app._resolve_path,
            app_dir=app.APP_DIR,
        )
        assert app.generate_multi(
            "refusal_direction", "Who are you?", progress=lambda *args, **kwargs: None
        ) == ("mock reply", "mock reply")
        assert len(requests) == 2
        assert requests[0]["steering"] is False
        assert requests[1]["steering"] == multi
        assert multi["conflict"] == "sequential"
        assert len(multi["vectors"]) == 4
        assert [vector["apply"]["prompt_positions"] for vector in multi["vectors"]] == [
            [-1],
            [-2],
            [-3],
            [-4],
        ]
        assert all(
            vector["source"].startswith("/remote/hf-space/")
            for vector in multi["vectors"]
        )


@pytest.mark.parametrize(
    "preset",
    sorted((SPACE / "configs").glob("*/*.json")),
    ids=lambda path: path.stem,
)
def test_bundled_preset_matches_engine_without_mutation(steering_spec, preset):
    config = json.loads(preset.read_text())
    original = deepcopy(config)
    assert isinstance(config["instruction"], str)
    assert all(type(value) in (int, float) for value in config["sampling"].values())
    assert all(
        isinstance(vector["layers"], list) for vector in config["steering"]["vectors"]
    )

    wire = runtime.load_steering_spec(
        config, lambda path: str(SPACE / path), app_dir=SPACE
    )
    steering_spec.model_validate(wire)
    overridden = runtime.load_steering_spec(
        config, lambda path: str(SPACE / path), app_dir=SPACE, scale_override=0
    )
    assert overridden["vectors"][0]["scale"] == 0
    assert overridden["vectors"][1:] == wire["vectors"][1:]
    steering_spec.model_validate(overridden)
    assert config == original
