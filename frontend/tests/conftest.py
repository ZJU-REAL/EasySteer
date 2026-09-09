"""Load the real payload code without bootstrapping the GPU engine."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def payload_modules(monkeypatch):
    for name in ("vllm", "vllm.model_hooks", "vllm.model_hooks.steering", "easysteer"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    payloads = load("vllm.model_hooks.steering.payloads", ROOT / "vllm-steer/vllm/model_hooks/steering/payloads.py")
    vectors = load("easysteer.vectors", ROOT / "easysteer/vectors.py")
    return payloads, vectors
