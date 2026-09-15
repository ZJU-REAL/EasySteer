# SPDX-License-Identifier: Apache-2.0
"""Canonical package imports stay lazy and legacy imports share definitions."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_script(script):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(ROOT / "vllm-steer"), str(ROOT))),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_discovery_does_not_import_optional_dependencies():
    run_script("""
        import importlib.abc
        import sys

        class NoDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, *args):
                if fullname.split(".")[0] in (
                    "numpy", "torch", "transformers", "sklearn", "vllm", "gguf"
                ):
                    raise AssertionError("Unexpected import: " + fullname)

        sys.meta_path.insert(0, NoDependencies())
        import easysteer
        from easysteer import extraction

        assert easysteer.__all__ == ["capture", "extraction", "training", "vectors"]
        assert all(name in dir(easysteer) for name in easysteer.__all__)
        assert "PCAExtractor" in dir(extraction)
        assert not hasattr(easysteer, "reft")
        assert not hasattr(extraction, "missing")
    """)


def test_diffmean_dispatch_only_loads_its_own_dependencies():
    run_script("""
        import importlib.abc
        import sys
        import numpy as np

        class NoHeavyDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, *args):
                if fullname.split(".")[0] in (
                    "torch", "transformers", "sklearn", "vllm", "gguf", "requests"
                ):
                    raise AssertionError("Unexpected import: " + fullname)

        sys.meta_path.insert(0, NoHeavyDependencies())
        from easysteer.extraction import extract_diffmean_control_vector

        result = extract_diffmean_control_vector(
            [[[[3., 2.]]], [[[1., 2.]]]], [0], [1]
        )
        np.testing.assert_array_equal(result.directions[0], [1., 0.])
        assert result.method == "diffmean"
        assert "easysteer.extraction.pca" not in sys.modules
    """)


def test_legacy_extraction_imports_share_canonical_definitions():
    run_script("""
        import importlib
        import pickle
        import warnings
        from easysteer import extraction

        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always", DeprecationWarning)
            from easysteer import steer
        assert any("easysteer.extraction" in str(m.message) for m in messages)
        assert steer.StatisticalControlVector is extraction.StatisticalControlVector

        modules = {
            "accumulators": "accumulators", "base_extractor": "base",
            "diffmean": "diffmean", "iti": "iti", "lat": "lat",
            "linear_probe": "linear_probe", "pca": "pca", "sae": "sae",
            "unified_interface": "api",
        }
        for old, new in modules.items():
            assert importlib.import_module("easysteer.steer." + old) is (
                importlib.import_module("easysteer.extraction." + new)
            )
        from easysteer.steer.utils import extract_token_hiddens
        assert extract_token_hiddens is extraction.extract_token_hiddens
        # Historical pickles resolve their old class path to the canonical type.
        assert pickle.loads(
            b"ceasysteer.steer.utils\\nStatisticalControlVector\\n."
        ) is extraction.StatisticalControlVector
    """)


def test_legacy_capture_imports_share_canonical_definitions():
    pytest.importorskip("torch")
    run_script("""
        import importlib
        import warnings
        from easysteer import capture

        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always", DeprecationWarning)
            from easysteer import hidden_states
        assert any("easysteer.capture" in str(m.message) for m in messages)
        assert hidden_states.CaptureResult is capture.CaptureResult
        assert hidden_states.capture is capture.capture
        assert hidden_states.capture_batches is capture.capture_batches
        assert importlib.import_module("easysteer.hidden_states.capture_result") is (
            importlib.import_module("easysteer.capture.api")
        )
    """)
