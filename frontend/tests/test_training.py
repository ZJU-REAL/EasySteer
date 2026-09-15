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
    for name in ("vllm", "vllm.model_hooks", "vllm.model_hooks.selection"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("schema", "spec"):
        qualified = f"vllm.model_hooks.selection.{name}"
        source = FRONTEND.parent / f"vllm-steer/vllm/model_hooks/selection/{name}.py"
        spec = importlib.util.spec_from_file_location(qualified, source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
    runtime = ModuleType("core.runtime")
    runtime.resource_manager = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "core.runtime", runtime)
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.setitem(
        sys.modules,
        "transformers.trainer_callback",
        SimpleNamespace(TrainerCallback=object),
    )
    train = ModuleType("easysteer.training")
    calls = []
    train.train = lambda **kwargs: calls.append(kwargs)

    def load_checkpoint(path):
        call = calls[-1]
        assert path == call["save_dir"]
        return SimpleNamespace(
            config=SimpleNamespace(prompt_template="Training prompt: %s\nAnswer:"),
            to_spec=lambda: SimpleNamespace(
                model_dump=lambda **kwargs: {
                    "vectors": [{
                        "data": {"version": 1, "kind": "checkpoint-payload"},
                        "algorithm": call["algorithm"],
                        "apply": call["apply"],
                    }],
                }
            )
        )

    train.load_checkpoint = load_checkpoint
    for name in ("easysteer",):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "easysteer.training", train)
    spec = importlib.util.spec_from_file_location(
        "training_backend", FRONTEND / "training_api.py"
    )
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
    (call,) = calls
    assert call["model_path"] == config["model_path"]
    assert call["examples"] == config["training_examples"]
    assert call["algorithm"] == config["algorithm"]
    assert call["save_dir"] == call["output_dir"] == config["output_dir"]
    assert {key: call[key] for key in config["steering_config"]} == config[
        "steering_config"
    ]
    assert {key: call[key] for key in config["training_args"]} == config[
        "training_args"
    ]
    assert call["apply"] == {"prompt_positions": [-1]}
    status = client.get("/api/train-status").json
    assert status["is_training"] is False
    assert status["error_message"] == ""
    assert status["status_message"].startswith("Training complete!")
    assert status["result"]["output_dir"] == config["output_dir"]
    assert status["result"]["model_path"] == config["model_path"]
    assert status["result"]["prompt_template"] == "Training prompt: %s\nAnswer:"
    (vector,) = status["result"]["steering"]["vectors"]
    assert vector["algorithm"] == config["algorithm"]
    assert vector["apply"] == call["apply"]
    assert vector["data"] == {"version": 1, "kind": "checkpoint-payload"}


def test_training_failure_and_log_retention(training, monkeypatch):
    module, client, trainer, _ = training

    def fail(**kwargs):
        (callback,) = kwargs["callbacks"]
        for step in range(110):
            callback.on_log(
                None, SimpleNamespace(global_step=step, epoch=1), None, {"loss": step}
            )
        raise RuntimeError("checkpoint write failed")

    monkeypatch.setattr(trainer, "train", fail)
    config = client.get("/api/train-config/emoji_loreft").json
    assert client.post("/api/train", json=config).status_code == 200
    status = client.get("/api/train-status").json
    assert status["is_training"] is False
    assert status["current_step"] == 109
    assert len(status["logs"]) == 100
    assert "Loss: 10.0000" in status["logs"][0]
    assert "Loss: 109.0000" in status["logs"][-1]
    assert status["error_message"] == "checkpoint write failed"


@pytest.mark.parametrize(
    "examples", ['[["input", "output"]]', [], [["input", 1]], [["input"]]]
)
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


@pytest.mark.parametrize("legacy_field", ["intervention", "reft_config"])
def test_removed_training_fields_fail_before_dispatch(training, legacy_field):
    _, client, _, calls = training
    config = client.get("/api/train-config/emoji_loreft").json
    config[legacy_field] = "bias" if legacy_field == "intervention" else {}
    response = client.post("/api/train", json=config)
    assert response.status_code == 400
    assert "API was removed" in response.json["error"]
    assert not calls


@pytest.mark.parametrize(
    "component", ["block_output", "attention_heads", "router_logits"]
)
def test_unsupported_training_component_fails_before_dispatch(training, component):
    _, client, _, calls = training
    config = client.get("/api/train-config/emoji_loreft").json
    config["steering_config"]["component"] = component
    response = client.post("/api/train", json=config)
    assert response.status_code == 400
    assert "hidden_states" in response.json["error"]
    assert not calls


def test_training_preserves_prompt_and_generation_selection(training):
    _, client, _, calls = training
    config = client.get("/api/train-config/emoji_loreft").json
    selection = {
        "prompt_window": [-3, None],
        "exclude_prompt_positions": [-2],
        "generation_window": [0, 5],
        "exclude_generation_tokens": [42],
    }
    config["steering_config"]["apply"] = selection
    response = client.post("/api/train", json=config)
    assert response.status_code == 200
    assert calls[0]["apply"] == selection
    assert response.json["steering_config"]["apply"] == selection


@pytest.mark.parametrize(
    "selection",
    [
        {},
        [],
        {"positions": [-1]},
        {"generation_positions": [-1]},
        {"prompt": "all", "generation_window": [2, 1]},
    ],
)
def test_invalid_selection_fails_before_dispatch(training, selection):
    _, client, _, calls = training
    config = client.get("/api/train-config/emoji_loreft").json
    config["steering_config"]["apply"] = selection
    response = client.post("/api/train", json=config)
    assert response.status_code == 400
    assert "Invalid training apply selection" in response.json["error"]
    assert not calls


@pytest.mark.parametrize("gpu", ["0,1", "", "-1", [0, 1]])
def test_web_training_rejects_multiple_or_missing_gpus(training, gpu):
    _, client, _, calls = training
    config = client.get("/api/train-config/emoji_loreft").json
    config["gpu_devices"] = gpu
    response = client.post("/api/train", json=config)
    assert response.status_code == 400
    assert "one GPU" in response.json["error"]
    assert "torchrun" in response.json["error"]
    assert not calls
