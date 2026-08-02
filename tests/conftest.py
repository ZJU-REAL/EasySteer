# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the EasySteer validation suites.

Engine-per-module convention (vLLM-style): a module that needs a GPU
engine declares module-level ``ENGINE_KWARGS`` and uses the ``llm``
fixture; every test in the module shares that one engine. Run each
module in its own process (see run_suites.sh) — vLLM engines do not
reliably release GPU state within a process.

Environment (same variables as the legacy scripts):
  GPU_ID                GPU to run on (default 0)
  STEER_TEST_MODEL      dense model path
  STEER_TEST_VECTOR     dense steering vector (gguf)
  STEER_TEST_MOE_MODEL  MoE model path (OLMoE)
  STEER_TEST_QWEN3      Qwen3-MoE model path
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Must be set before vllm/torch import (conftest imports first).
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU_ID", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
_TRACE_DIR = tempfile.mkdtemp(prefix="steer_trace_pytest_")
os.environ.setdefault("VLLM_STEER_TRACE_DIR", _TRACE_DIR)

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def llm(request):
    """One engine per module, built from the module's ENGINE_KWARGS."""
    from vllm import LLM

    kwargs = getattr(request.module, "ENGINE_KWARGS", None)
    assert kwargs is not None, (
        f"{request.module.__name__} uses the llm fixture but declares no "
        "ENGINE_KWARGS"
    )
    engine = LLM(**dict(kwargs))
    yield engine


@pytest.fixture()
def trace(llm):
    """Steering-trace oracle bound to the module engine."""
    from helpers import TraceOracle

    return TraceOracle(llm, os.environ["VLLM_STEER_TRACE_DIR"])
