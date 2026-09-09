"""The API Space must work without torch or vLLM installed."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPACE = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_module("space_runtime", SPACE / "runtime.py")
exporter = load_module("space_exporter", SPACE / "export_payload.py")


def test_default_api_mode_requires_configuration():
    with pytest.raises(ValueError, match="VLLM_API_URL, VLLM_MODEL_NAME"):
        runtime.demo_mode({})
    assert runtime.demo_mode({"VLLM_API_URL": "http://server/v1", "VLLM_MODEL_NAME": "model"}) == "api"
    assert runtime.demo_mode({"DEMO_MODE": "gpu"}) == "gpu"
    for invalid in ("typo", "auto"):
        with pytest.raises(ValueError, match="DEMO_MODE"):
            runtime.demo_mode({"DEMO_MODE": invalid})


def test_export_reports_lfs_pointer_before_importing_engine(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
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
    direct = app.build_single_spec_wire(app.SINGLE_CONFIGS["emotion_direct"], app._vector_source)
    assert direct["vectors"][0]["source"].startswith("/remote/hf-space/")
    reft = app.build_single_spec_wire(app.SINGLE_CONFIGS["emoji_loreft"], app._vector_source)
    vector = reft["vectors"][0]
    assert vector["data"]["kind"] == "reft"
    assert vector["data"]["extra"]["layer"] == 22
    assert "source" not in vector
    assert vector["apply"] == {"prompt_positions": [-1]}
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="mock reply"))])

    monkeypatch.setattr(app._api_client.chat.completions, "create", create)
    assert app.generate_single("emoji_loreft", "Who are you?", 1.0, progress=lambda *args, **kwargs: None) == ("mock reply", "mock reply")
    assert len(requests) == 2
    assert requests[0]["extra_body"]["steering"] is False
    assert requests[1]["extra_body"]["steering"] == reft
