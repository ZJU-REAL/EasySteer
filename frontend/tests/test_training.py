"""Exercise the preset -> HTTP -> shared training boundary without training."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from flask import Flask

FRONTEND = Path(__file__).resolve().parents[1]


@pytest.fixture
def training(monkeypatch):
    monkeypatch.syspath_prepend(str(FRONTEND))
    runtime = ModuleType("core.runtime")
    runtime.resource_manager = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "core.runtime", runtime)
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.trainer_callback", SimpleNamespace(TrainerCallback=object))
    train = ModuleType("easysteer.reft.train")
    calls = []
    train.train_reft = lambda **kwargs: calls.append(kwargs)
    for name in ("easysteer", "easysteer.reft"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "easysteer.reft.train", train)
    spec = importlib.util.spec_from_file_location("training_backend", FRONTEND / "training_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Thread:
        def __init__(self, target):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(module.threading, "Thread", Thread)
    app = Flask(__name__)
    app.register_blueprint(module.training_bp)
    return module, app.test_client(), train, calls


@pytest.mark.parametrize("preset", ["emoji_bias", "emoji_loreft"])
def test_preset_request_preserves_shared_training_arguments(training, preset):
    module, client, _, calls = training
    config = client.get(f"/api/train-config/{preset}").json
    response = client.post("/api/train", json=config)
    assert response.status_code == 200
    assert response.json["training_examples_count"] == len(config["training_examples"])
    call, = calls
    assert call["model_path"] == config["model_path"]
    assert call["examples"] == config["training_examples"]
    assert call["intervention"] == config["intervention"]
    assert call["save_dir"] == call["output_dir"] == config["output_dir"]
    assert {key: call[key] for key in config["reft_config"]} == config["reft_config"]
    assert {key: call[key] for key in config["training_args"]} == config["training_args"]
    status = client.get("/api/train-status").json
    assert status["is_training"] is False
    assert status["error_message"] == ""
    assert status["status_message"].startswith("Training complete!")


def test_training_failure_and_log_retention(training, monkeypatch):
    module, client, trainer, _ = training

    def fail(**kwargs):
        callback, = kwargs["callbacks"]
        for step in range(110):
            callback.on_log(None, SimpleNamespace(global_step=step, epoch=1), None, {"loss": step})
        raise RuntimeError("checkpoint write failed")

    monkeypatch.setattr(trainer, "train_reft", fail)
    config = client.get("/api/train-config/emoji_loreft").json
    assert client.post("/api/train", json=config).status_code == 200
    status = client.get("/api/train-status").json
    assert status["is_training"] is False
    assert status["current_step"] == 109
    assert len(status["logs"]) == 100
    assert "Loss: 10.0000" in status["logs"][0]
    assert "Loss: 109.0000" in status["logs"][-1]
    assert status["error_message"] == "checkpoint write failed"


@pytest.mark.parametrize("examples", ['[["input", "output"]]', [], [["input", 1]], [["input"]]])
def test_invalid_training_examples_fail_before_dispatch(training, examples):
    _, client, _, calls = training
    config = client.get("/api/train-config/emoji_loreft").json
    config["training_examples"] = examples
    assert client.post("/api/train", json=config).status_code == 400
    assert not calls


def test_nested_output_dir_is_not_an_implicit_request_format(training):
    _, client, _, calls = training
    config = client.get("/api/train-config/emoji_loreft").json
    config["training_args"]["output_dir"] = config.pop("output_dir")
    assert client.post("/api/train", json=config).status_code == 400
    assert not calls
