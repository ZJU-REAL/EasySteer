# SPDX-License-Identifier: Apache-2.0
"""Check public examples against the steering schema without loading a model.

This catches stale explicit keyword arguments in notebook code cells and Python
fences. It does not execute examples or inspect their historical outputs.
"""

import ast
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC_NAMES = {"SelectSpec", "ApplySpec", "VectorSpec", "SteeringSpec"}


def _schema_fields():
    from vllm.steer_vectors.api import ApplySpec, SelectSpec, SteeringSpec, VectorSpec

    return {
        cls.__name__: set(cls.model_fields)
        for cls in (SelectSpec, ApplySpec, VectorSpec, SteeringSpec)
    }


def _examples():
    for folder in ("experiment", "replications", "easysteer/reft/pyreft/examples"):
        for path in sorted((ROOT / folder).rglob("*.ipynb")):
            for index, cell in enumerate(json.loads(path.read_text())["cells"]):
                if cell["cell_type"] == "code":
                    # Notebook shell/magic lines do not call the Python API.
                    source = "".join(
                        line
                        for line in cell["source"]
                        if not line.lstrip().startswith(("!", "%"))
                    )
                    yield f"{path.relative_to(ROOT)} cell {index}", source
    docs = [ROOT / "README.md", ROOT / "README_zh.md"]
    for folder in ("docs", "replications", "experiment"):
        docs.extend(sorted((ROOT / folder).rglob("*.md")))
    for path in docs:
        for match in re.finditer(
            r"^[ \t]*```python[ \t]*\n(.*?)^[ \t]*```[ \t]*$",
            path.read_text(),
            re.MULTILINE | re.DOTALL,
        ):
            line = path.read_text()[: match.start()].count("\n") + 1
            yield f"{path.relative_to(ROOT)}:{line}", textwrap.dedent(match[1])


def test_example_steering_keywords_match_current_schema():
    fields = _schema_fields()
    checked = 0
    for location, source in _examples():
        tree = ast.parse(source, filename=location)
        aliases = {name: name for name in SPEC_NAMES}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                "vllm.steer_vectors",
                "vllm.model_hooks.steering.api",
                "vllm.model_hooks.selection.spec",
                "vllm.steer_vectors.api",
                "vllm.capture",
            }:
                aliases.update(
                    {
                        item.asname or item.name: item.name
                        for item in node.names
                        if item.name in SPEC_NAMES
                    }
                )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = aliases.get(node.func.id)
            if name is None:
                continue
            unknown = {kw.arg for kw in node.keywords if kw.arg} - fields[name]
            assert not unknown, (
                f"{location}:{node.lineno}: {name} unknown fields {unknown}"
            )
            checked += 1
    assert checked, "No steering examples found; check the example discovery paths"
