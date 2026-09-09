"""CPU checks for UI routing and the training demo's HTTP boundary."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from flask import Blueprint

FRONTEND = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_built_ui_and_job_api_share_origin(monkeypatch, tmp_path):
    # Stub only GPU/job imports; exercise the real Flask routing and serving.
    monkeypatch.syspath_prepend(str(FRONTEND))
    for name, attribute in (
        ("training_api", "training_bp"),
        ("extraction_api", "extraction_bp"),
        ("sae_api", "sae_bp"),
    ):
        module = ModuleType(name)
        setattr(module, attribute, Blueprint(name, name))
        monkeypatch.setitem(sys.modules, name, module)
    runtime = ModuleType("core.runtime")
    runtime.llm_manager = SimpleNamespace(_instances={})
    monkeypatch.setitem(sys.modules, "core.runtime", runtime)
    module = load_module("frontend_deployment", FRONTEND / "app.py")
    (tmp_path / "index.html").write_text("<html>EasySteer UI</html>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("// UI bundle")
    monkeypatch.setattr(module, "UI_DIR", tmp_path)
    client = module.app.test_client()
    assert b"EasySteer UI" in client.get("/").data
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/api/health").json["status"] == "healthy"
    assert client.get("/api/missing").status_code == 404
    assert client.get("/../app.py").status_code == 404


@pytest.mark.parametrize("kind,algorithm", [("reft", "loreft"), ("direction", "direct")])
def test_training_demo_uses_openai_payload(monkeypatch, payload_modules, kind, algorithm):
    module = load_module("training_demo", FRONTEND / "demo_training.py")
    payloads, vectors = payload_modules
    payload = (payloads.ReftIntervention([[1]], [[1]], layer=8) if kind == "reft"
               else payloads.DirectionVector({8: [1]}))
    monkeypatch.setattr(vectors, "from_pyreft", lambda path: payload)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "test reply"}}]},
        )

    monkeypatch.setattr(module.requests, "post", post)
    module.test_inference("served-model", "/checkpoint", ["Hello"], "http://example/v1/", "key")
    url, kwargs = calls[0]
    assert url == "http://example/v1/chat/completions"
    assert kwargs["headers"] == {"Authorization": "Bearer key"}
    assert kwargs["json"]["model"] == "served-model"
    vector = kwargs["json"]["steering"]["vectors"][0]
    assert vector["algorithm"] == algorithm
    assert next(iter(vector["data"]["tensors"].values()))["data"] == "AACAPw=="
    assert vector["apply"] == {"prompt_positions": [-1]}


def test_training_demo_sends_the_current_job_request(monkeypatch):
    module = load_module("training_demo", FRONTEND / "demo_training.py")
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return SimpleNamespace(status_code=200, json=lambda: {"message": "started"})

    monkeypatch.setattr(module.requests, "post", post)
    monkeypatch.setattr(module, "monitor_training", lambda: True)
    assert module.start_training_demo("model") is True
    request, = calls
    assert request["output_dir"] == "./results/demo_emoji_training"
    assert request["intervention"] == "loreft"
    assert isinstance(request["training_examples"], list)
    assert "output_dir" not in request["training_args"]


def test_gunicorn_restart_exits_worker_instead_of_starting_another_master(monkeypatch):
    module = load_module("job_resources", FRONTEND / "core" / "resource_manager.py")
    manager = module.ResourceManager
    monkeypatch.setenv("SERVER_SOFTWARE", "gunicorn/21.2.0")
    monkeypatch.setattr(manager, "cleanup_all_resources", lambda: {})
    monkeypatch.setattr(module.time, "sleep", lambda delay: None)
    callbacks = []

    class Thread:
        def __init__(self, target):
            callbacks.append(target)

        def start(self):
            pass

    monkeypatch.setattr(module.threading, "Thread", Thread)
    manager.restart_backend(delay=0)

    def exit_worker(code):
        assert code == 0
        raise SystemExit(code)

    monkeypatch.setattr(module.os, "_exit", exit_worker)
    monkeypatch.setattr(module.os, "execv", lambda *args: pytest.fail("started a second master"))
    with pytest.raises(SystemExit):
        callbacks[0]()


@pytest.fixture
def extraction(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(FRONTEND))
    runtime = ModuleType("core.runtime")
    engine = object()
    activity = []
    calls = []

    def load_engine(**kwargs):
        activity.append("load")
        return engine

    def cuda_available():
        activity.append("cuda")
        return False

    runtime.llm_manager = SimpleNamespace(get_or_create_llm=load_engine)
    runtime.resource_manager = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "core.runtime", runtime)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=cuda_available)))
    captured = SimpleNamespace(layer_ids=[8, 22])
    hidden_states = ModuleType("easysteer.hidden_states")

    def capture(llm, prompts, max_tokens=1):
        assert llm is engine
        activity.append("capture")
        calls.append((prompts, max_tokens))
        return captured

    hidden_states.capture = capture
    monkeypatch.setitem(sys.modules, "easysteer", ModuleType("easysteer"))
    monkeypatch.setitem(sys.modules, "easysteer.hidden_states", hidden_states)
    extracted = []

    def extract_statistical_control_vector(method, all_hidden_states, positive_indices, negative_indices=None, **kwargs):
        activity.append("extract")
        extracted.append(dict(method=method, all_hidden_states=all_hidden_states,
                              positive_indices=positive_indices, negative_indices=negative_indices, **kwargs))
        return SimpleNamespace(
            directions={8: [1], 22: [2]}, metadata={},
            export_gguf=lambda output: Path(output).write_bytes(b"mock vector"),
        )

    steer = ModuleType("easysteer.steer")
    steer.extract_statistical_control_vector = extract_statistical_control_vector
    monkeypatch.setitem(sys.modules, "easysteer.steer", steer)
    module = load_module("extraction_backend", FRONTEND / "extraction_api.py")
    monkeypatch.chdir(tmp_path)
    config = {
        "model_path": "model", "gpu_devices": "0", "method": "diffmean",
        "positive_samples": ["happy"], "negative_samples": ["sad"],
        "output_path": "vector.gguf",
    }
    return SimpleNamespace(module=module, config=config, calls=calls, extracted=extracted,
                           captured=captured, activity=activity, output=tmp_path / "vector.gguf")


@pytest.mark.parametrize("method,token_pos,expected", [
    ("diffmean", -1, -1), ("lat", " -2 ", -2), ("pca", 0, 0), ("diffmean", "+2", 2),
])
def test_extraction_uses_labelled_capture_and_shared_dispatch(extraction, method, token_pos, expected):
    job = extraction
    job.config.update(method=method, token_pos=token_pos)
    job.module.run_extraction(job.config)
    assert job.calls == [(["happy", "sad"], 1)]
    call, = job.extracted
    assert call["all_hidden_states"] is job.captured
    assert call["positive_indices"] == [0]
    assert call["negative_indices"] == [1]
    assert call["method"] == method
    assert call["token_pos"] == expected
    assert call["normalize"] is True
    assert job.module.extraction_status["result"]["layers_extracted"] == 2
    assert job.module.extraction_status["result"]["method"] == method
    assert job.module.extraction_status["is_extracting"] is False
    assert job.output.exists()


def test_extraction_default_position_remains_last_token(extraction):
    extraction.module.run_extraction(extraction.config)
    assert extraction.extracted[0]["token_pos"] == -1


@pytest.mark.parametrize("token_pos", [True, False, 0.5, 1.0, "1.5", "1_0", "bad", "", None, []])
def test_invalid_token_position_fails_before_cuda_or_capture(extraction, token_pos):
    job = extraction
    job.config["token_pos"] = token_pos
    job.module.run_extraction(job.config)
    assert "token_pos must be an integer position" in job.module.extraction_status["error_message"]
    assert job.module.extraction_status["is_extracting"] is False
    assert not job.activity
    assert not job.output.exists()


@pytest.mark.parametrize("changes,error", [
    ({"gpu_devices": "0,1"}, "one GPU"),
    ({"method": "linear_probe"}, "Unsupported extraction method"),
])
def test_unsupported_job_configuration_fails_before_model_load(extraction, changes, error):
    job = extraction
    job.config.update(changes)
    job.module.run_extraction(job.config)
    assert error in job.module.extraction_status["error_message"]
    assert not job.activity
    assert not job.output.exists()
