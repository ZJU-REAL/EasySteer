# SPDX-License-Identifier: Apache-2.0
"""Check public API callers from source, without installing torch or vLLM.

This covers explicit imports and keyword arguments, including notebook cells
and Markdown fences. Numerical and dynamically generated examples have separate
runtime tests; this check does not execute a model or historical outputs.
"""

import ast
import json
import re
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC_NAMES = {"SelectSpec", "ApplySpec", "VectorSpec", "SteeringSpec"}
PUBLIC_FOLDERS = (
    "experiment",
    "replications",
    "examples",
    "frontend",
    "hf-space",
    "docker",
    "tools",
)
DEPRECATED_MODULES = ("easysteer.hidden_states", "easysteer.steer", "easysteer.reft")


def _definition(path, *names):
    node = ast.parse((ROOT / path).read_text())
    for name in names:
        node = next(
            child for child in node.body if getattr(child, "name", None) == name
        )
    return node


def _fields(node):
    return {
        child.target.id
        for child in node.body
        if isinstance(child, ast.AnnAssign)
        and isinstance(child.target, ast.Name)
        and not child.target.id.startswith("_")
    }


def _parameters(path, *names):
    args = _definition(path, *names).args
    return {arg.arg for arg in (*args.args, *args.kwonlyargs)} - {"self", "cls"}


def _public_exports():
    modules = {}
    for name in ("capture", "extraction", "training", "vectors"):
        path = (
            f"easysteer/{name}.py"
            if name == "vectors"
            else f"easysteer/{name}/__init__.py"
        )
        declaration = "_EXPORTS" if name == "extraction" else "__all__"
        assignment = next(
            node
            for node in _definition(path).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == declaration
                for target in node.targets
            )
        )
        modules[f"easysteer.{name}"] = set(ast.literal_eval(assignment.value))
    return modules


def _schema_fields():
    fields = {
        name: _fields(_definition("vllm-steer/vllm/model_hooks/steering/api.py", name))
        for name in ("VectorSpec", "SteeringSpec")
    }
    fields["SelectSpec"] = fields["ApplySpec"] = _fields(
        _definition("vllm-steer/vllm/model_hooks/selection/spec.py", "SelectSpec")
    )
    return fields


def _call_fields():
    fields = _schema_fields()
    sampling = _fields(
        _definition("vllm-steer/vllm/sampling_params.py", "SamplingParams")
    )
    fields["capture"] = _parameters("easysteer/capture/api.py", "capture") | sampling
    fields["capture_batches"] = (
        _parameters("easysteer/capture/api.py", "capture_batches") | fields["capture"]
    ) - {"sample_indices"}
    extraction_options = set()
    for module, cls in (
        ("diffmean", "DiffMeanExtractor"),
        ("pca", "PCAExtractor"),
        ("lat", "LATExtractor"),
        ("linear_probe", "LinearProbeExtractor"),
        ("iti", "ITIExtractor"),
    ):
        extraction_options |= _parameters(
            f"easysteer/extraction/{module}.py", cls, "extract"
        )
    # The stream wrapper owns training labels; ITI may also receive validation labels.
    extraction_options -= {"all_hidden_states", "positive_indices", "negative_indices"}
    fields["extract"] = (
        _parameters("easysteer/extraction/streaming.py", "extract")
        | extraction_options
        | {"validation_labels"}
    )
    fields["TrainingConfig"] = _fields(
        _definition("easysteer/training/checkpoint.py", "TrainingConfig")
    )
    return fields


def _python_fences(text):
    for match in re.finditer(
        r"^[ \t]*```(?:python|py)[ \t]*\n(.*?)^[ \t]*```[ \t]*$",
        text,
        re.MULTILINE | re.DOTALL,
    ):
        line = text[: match.start()].count("\n") + 1
        yield line, textwrap.dedent(match[1])


def _examples():
    for folder in PUBLIC_FOLDERS:
        for path in sorted((ROOT / folder).rglob("*")):
            if any(
                part in {"node_modules", "dist", "__pycache__"} for part in path.parts
            ):
                continue
            if (
                path.suffix == ".py"
                and "tests" not in path.relative_to(ROOT / folder).parts
            ):
                yield str(path.relative_to(ROOT)), path.read_text()
            if path.suffix != ".ipynb":
                continue
            code = []
            for index, cell in enumerate(json.loads(path.read_text())["cells"]):
                source = cell["source"]
                if not isinstance(source, str):
                    source = "".join(source)
                if cell["cell_type"] == "code":
                    # Notebook shell/magic lines do not call the Python API.
                    source = "".join(
                        line + "\n"
                        for line in source.splitlines()
                        if not line.lstrip().startswith(("!", "%"))
                    )
                    code.append(f"# Cell {index}\n{source}")
                elif cell["cell_type"] == "markdown":
                    for line, example in _python_fences(source):
                        yield f"{path.relative_to(ROOT)} cell {index}:{line}", example
            # Imports in earlier cells also apply to later calls.
            yield str(path.relative_to(ROOT)), "\n".join(code)
    docs = [ROOT / "README.md", ROOT / "README_zh.md"]
    for folder in ("docs", *PUBLIC_FOLDERS):
        docs.extend(
            path
            for path in sorted((ROOT / folder).rglob("*.md"))
            if not any(part in {"node_modules", "dist"} for part in path.parts)
        )
    docs.append(ROOT / "vllm-steer/docs/features/steer_vectors.md")
    for path in docs:
        examples = [
            f"# Line {line}\n{source}"
            for line, source in _python_fences(path.read_text())
        ]
        if examples:
            yield str(path.relative_to(ROOT)), "\n".join(examples)


def _qualified_name(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{_qualified_name(node.value, aliases)}.{node.attr}"
    return ""


def _api_errors(source, fields, exports=None):
    tree = ast.parse(source)
    aliases = {name: f"vllm.steer_vectors.{name}" for name in SPEC_NAMES}
    errors = []
    for node in ast.walk(tree):
        imports = {}
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            imports = {
                item.asname or item.name: f"{node.module}.{item.name}"
                for item in node.names
            }
            if exports and node.module in exports:
                unknown = {item.name for item in node.names} - exports[node.module]
                if unknown:
                    errors.append(
                        f"line {node.lineno}: {node.module} does not export {sorted(unknown)}"
                    )
        elif isinstance(node, ast.Import):
            imports = {
                item.asname or item.name.split(".")[0]: item.name
                if item.asname
                else item.name.split(".")[0]
                for item in node.names
            }
        aliases.update(imports)
        imported = (
            [item.name for item in node.names]
            if isinstance(node, ast.Import)
            else imports.values()
        )
        for module in imported:
            if any(
                module == old or module.startswith(old + ".")
                for old in DEPRECATED_MODULES
            ):
                errors.append(f"line {node.lineno}: deprecated import {module}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qualified = _qualified_name(node.func, aliases)
        name = qualified.rsplit(".", 1)[-1]
        if name not in fields or not qualified.startswith(("vllm.", "easysteer.")):
            continue
        unknown = {kw.arg for kw in node.keywords if kw.arg} - fields[name]
        if unknown:
            errors.append(
                f"line {node.lineno}: {qualified} unknown fields {sorted(unknown)}"
            )
    return errors


def test_public_examples_use_current_api():
    fields = _call_fields()
    exports = _public_exports()
    checked = 0
    for location, source in _examples():
        errors = _api_errors(source, fields, exports)
        assert not errors, f"{location}:\n" + "\n".join(errors)
        checked += 1
    assert checked, "No examples found; check the discovery paths"


@pytest.mark.parametrize(
    "source, expected",
    [
        ("from easysteer import hidden_states as hs", "deprecated import"),
        ("import easysteer.steer", "deprecated import"),
        ("from easysteer.reft.train import train_reft", "deprecated import"),
        (
            "from easysteer.capture import get_all_hidden_states_generate",
            "does not export",
        ),
        ("from easysteer.training import train_reft", "does not export"),
        (
            "import easysteer.capture as hs\nhs.capture(llm, prompts, positions=[-1])",
            "positions",
        ),
        (
            "from easysteer.capture import capture_batches as batches\nbatches(llm, prompts, reduce='mean')",
            "reduce",
        ),
        (
            "from vllm.steer_vectors import ApplySpec as Apply\nApply(phases=['prompt'])",
            "phases",
        ),
        (
            "from easysteer.extraction import extract\nextract(rows, labels, positive_indices=[0])",
            "positive_indices",
        ),
    ],
)
def test_api_audit_detects_removed_calls(source, expected):
    assert any(
        expected in error
        for error in _api_errors(source, _call_fields(), _public_exports())
    )
