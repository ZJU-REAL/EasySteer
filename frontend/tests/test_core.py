"""CPU regression checks for model reuse and the lightweight core boundary."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

FRONTEND = Path(__file__).resolve().parents[1]


def test_light_utilities_do_not_initialize_runtime():
    code = """
import sys
from core import ConfigStore, get_message, require_fields
assert ConfigStore('training').get('emoji_bias')['intervention'] == 'bias'
assert not {'core.runtime', 'core.llm_manager', 'vllm', 'torch', 'transformers'} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True,
        env={**os.environ, "PYTHONPATH": str(FRONTEND)}, capture_output=True, text=True,
    )


@pytest.fixture
def manager(monkeypatch):
    created = []

    def llm(**config):
        instance = SimpleNamespace(config=config)
        created.append(instance)
        return instance

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=llm))
    spec = importlib.util.spec_from_file_location("test_llm_manager", FRONTEND / "core/llm_manager.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LLMManager(), created


@pytest.mark.parametrize("changed", [
    {"model_path": "other"}, {"gpu_devices": "1"}, {"enforce_eager": False},
    {"enable_chunked_prefill": True}, {"enable_chunked_prefill": False},
    {"enable_prefix_caching": True}, {"enable_prefix_caching": False},
    {"enable_steer_vector": True}, {"dtype": "bfloat16"},
    {"max_model_len": 4096}, {"tensor_parallel_size": 2},
    {"compilation_config": {"mode": 0}},
])
def test_changed_constructor_arguments_never_reuse_an_engine(manager, changed):
    cache, created = manager
    first = cache.get_or_create_llm("model")
    second = cache.get_or_create_llm(**{"model_path": "model", **changed})
    assert first is not second
    assert len(created) == 2


def test_equivalent_effective_config_reuses_engine_and_survives_input_mutation(manager):
    cache, created = manager
    nested = {"mode": 0, "extra": {"a": [1, 2], "b": True}}
    first = cache.get_or_create_llm("ignored", model="model", gpu_devices=" 0, 1 ", compilation_config=nested)
    reordered = {"extra": {"b": True, "a": [1, 2]}, "mode": 0}
    assert cache.get_or_create_llm("model", gpu_devices="0,1", compilation_config=reordered) is first
    nested["extra"]["a"].append(3)
    assert cache.get_or_create_llm("model", gpu_devices="0,1", compilation_config=nested) is not first
    assert len(created) == 2


def test_nonserializable_config_is_owned_but_never_reused(manager):
    cache, created = manager
    callback = lambda value: value
    assert cache.get_or_create_llm("model", hf_overrides=callback) is not cache.get_or_create_llm("model", hf_overrides=callback)
    assert len(created) == 2
    assert cache.clear_all_instances() == 2
    assert len(cache) == 0


def test_failed_construction_does_not_populate_cache(manager, monkeypatch):
    cache, _ = manager
    # Patch the defining function globals, without importing the real engine.
    def fail(**kwargs):
        raise RuntimeError("model unavailable")
    monkeypatch.setitem(cache.get_or_create_llm.__globals__, "LLM", fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        cache.get_or_create_llm("model")
    assert len(cache) == 0
