"""Packaged clients must agree with the engine's authoring contracts."""

from pathlib import Path
import runpy

from vllm.model_hooks.steering.algorithms.registry import ALGORITHM_REGISTRY
from vllm.model_hooks.steering.capabilities import ALGORITHM_CAPABILITIES


def test_builtin_algorithm_catalog_matches_registered_implementations():
    assert set(ALGORITHM_CAPABILITIES) == set(ALGORITHM_REGISTRY)


def test_packaged_client_contracts_are_current():
    script = Path(__file__).resolve().parents[2] / "tools/export_client_contracts.py"
    generate = runpy.run_path(str(script))["generated_files"]
    for path, content in generate():
        assert path.read_text() == content, f"Regenerate client contracts: {path.name}"
