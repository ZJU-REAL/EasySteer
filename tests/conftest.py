# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the EasySteer validation suites.

A module declares ``ENGINE_KWARGS`` and uses the ``llm`` fixture. Only
modules declaring the same ``ENGINE_PROFILE`` and identical kwargs may
share an engine; other configurations run in separate processes.
Trace is enabled before engine startup when selected tests need it.

Environment (same variables as the legacy scripts):
  GPU_ID                GPU to run on (default 0)
  STEER_TEST_MODEL      dense model path
  STEER_TEST_VECTOR     dense steering vector (gguf)
  STEER_TEST_MOE_MODEL  MoE model path (OLMoE)
  STEER_TEST_QWEN3      Qwen3-MoE model path
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Must be set before vllm/torch import (conftest imports first).
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU_ID", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import pytest  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--steer-extended", action="store_true",
        help="Run the full 51-scale sweep instead of the representative sweep.",
    )


def pytest_collection_modifyitems(config, items):
    # Direct readers declare STEER_TEST_TRACE; fixture users are discovered
    # through their fixture closure. Both are known before any LLM is created.
    needs_trace = any(
        "trace" in item.fixturenames
        or getattr(item.module, "STEER_TEST_TRACE", False)
        for item in items
    )
    if needs_trace and not os.environ.get("VLLM_STEER_TRACE_DIR"):
        os.environ["VLLM_STEER_TRACE_DIR"] = tempfile.mkdtemp(
            prefix="steer_trace_pytest_"
        )
    config._steer_engine_timings = []


def pytest_terminal_summary(terminalreporter, config):
    timings = getattr(config, "_steer_engine_timings", [])
    if not timings:
        return
    terminalreporter.section("steering engine startup (seconds)")
    for row in timings:
        terminalreporter.write_line(f"{row['seconds']:.3f} {row['engine']}")
    prefix = os.environ.get("STEER_TEST_ARTIFACT_PREFIX")
    if prefix:
        Path(prefix + ".engines.json").write_text(
            json.dumps(timings, indent=2) + "\n"
        )


@pytest.fixture(scope="session")
def shared_engines():
    """Only explicitly named, identical profiles may share a GPU engine."""
    return {}


@pytest.fixture(scope="module")
def llm(request, shared_engines):
    """One engine per module, built from the module's ENGINE_KWARGS."""
    from vllm import LLM

    kwargs = getattr(request.module, "ENGINE_KWARGS", None)
    assert kwargs is not None, (
        f"{request.module.__name__} uses the llm fixture but declares no "
        "ENGINE_KWARGS"
    )
    profile = getattr(request.module, "ENGINE_PROFILE", None)
    if profile in shared_engines:
        previous_kwargs, engine = shared_engines[profile]
        assert kwargs == previous_kwargs, f"Conflicting engine profile: {profile}"
    else:
        started = time.monotonic()
        engine = LLM(**dict(kwargs))
        request.config._steer_engine_timings.append({
            "engine": profile or request.module.__name__,
            "seconds": round(time.monotonic() - started, 3),
        })
        if profile is not None:
            shared_engines[profile] = (dict(kwargs), engine)
    yield engine


@pytest.fixture()
def trace(llm):
    """Steering-trace oracle bound to the module engine."""
    from helpers import TraceOracle

    return TraceOracle(llm, os.environ["VLLM_STEER_TRACE_DIR"])
