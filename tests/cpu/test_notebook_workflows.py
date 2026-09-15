# SPDX-License-Identifier: Apache-2.0
"""Exercise notebook estimators and row attribution without loading a model."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from easysteer.extraction import StatisticalControlVector, extract
from easysteer.extraction.api import extract_linear_probe_control_vector

ROOT = Path(__file__).resolve().parents[2]


def _cells_calling(path, name):
    """Locate executable workflow cells by a call, independent of cell order."""
    notebook = json.loads((ROOT / path).read_text())
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
            for node in ast.walk(ast.parse(source))
        ):
            cells.append(compile(source, f"{path}:cell-{index}", "exec"))
    assert cells, f"No {name} workflow found in {path}"
    return cells


class _Tensor(np.ndarray):
    def float(self):
        return self.astype(np.float32)

    def numpy(self):
        return np.asarray(self)


class _Capture:
    """CaptureResult's row protocol, with fetch order deliberately reversed."""

    component = "hidden_states"
    model = "fixture"

    def __init__(self, samples, *, sample_indices=None, positions=None):
        self.selection = {"prompt": "all"}
        self.samples = {
            layer: [np.asarray(rows, dtype=np.float32).view(_Tensor) for rows in values]
            for layer, values in samples.items()
        }
        self.layer_ids = sorted(samples)
        self.layers = {
            layer: np.concatenate(values[::-1]).view(_Tensor)
            for layer, values in self.samples.items()
        }
        self.sample_indices = (
            list(range(len(self))) if sample_indices is None else sample_indices
        )
        self.positions = positions

    def __len__(self):
        return len(self.samples[self.layer_ids[0]])

    def sample_rows(self, sample, layer):
        return self.samples[layer][sample]

    def sample_positions(self, sample):
        if self.positions is not None:
            return self.positions[sample]
        return list(range(len(self.sample_rows(sample, self.layer_ids[0]))))

    def token(self, sample, layer, position=-1):
        return self.sample_rows(sample, layer)[position]

    def rows(self, layer):
        return self.layers[layer]


@pytest.fixture
def exports(monkeypatch):
    vectors = {}

    def export(vector, path):
        vectors[str(path)] = vector

    monkeypatch.setattr(StatisticalControlVector, "export_gguf", export)
    return vectors


def test_hallucination_preserves_exact_estimators(exports):
    rng = np.random.default_rng(123)
    rows = rng.normal(size=(40, 4)).astype(np.float32)
    rows[:20, 0] += 2
    rows[20:, 0] -= 2
    captured = _Capture({14: list(rows[:, None, :])})
    for code in _cells_calling("experiment/hallucination/data.ipynb", "extract"):
        exec(  # noqa: S102 - Execute repository-owned notebook code.
            code,
            {
                "captured": captured,
                "labels": [True] * 20 + [False] * 20,
                "extract": extract,
                "EXTRACTION_BYTES": 1024**3,
                "llm": None,
                "release_capture_cache": lambda llm: None,
            },
        )

    gap = rows[:20].mean(axis=0) - rows[20:].mean(axis=0)
    pairs = (rows[:20] - rows[20:]) / 2
    _, _, components = np.linalg.svd(
        np.concatenate([pairs, -pairs]), full_matrices=False
    )
    pca = components[0]
    if pca @ gap < 0:
        pca = -pca
    probe = extract_linear_probe_control_vector(
        captured, list(range(20)), list(range(20, 40)), normalize=True
    )
    expected = {
        "caa": gap / np.linalg.norm(gap),
        "pca": pca,
        "probe": probe.directions[14],
    }
    assert len(exports) == 6
    for fold in (1, 2):
        for method, direction in expected.items():
            vector = exports[f"real{fold}-{method}.gguf"]
            np.testing.assert_allclose(vector.directions[14], direction, atol=1e-6)
            assert vector.component == "hidden_states"
        assert exports[f"real{fold}-pca.gguf"].metadata["variant"] == "center"


@pytest.mark.parametrize(
    "notebook,count_name,normalize",
    [
        ("fractreason", "problems", True),
        ("controlingthinkingspeed", "pairs", False),
    ],
)
def test_full_math_pair_sets_pass_exact_pca_budget(
    notebook, count_name, normalize, monkeypatch, exports
):
    class ShapeCapture:
        def __len__(self):
            return len(self.layers[0])

        def sample_rows(self, *args):
            pytest.fail("Budget validation must not read activation rows")

    captured = ShapeCapture()
    captured.layer_ids = list(range(28))
    # Represent 1,000 rows per layer without allocating activation matrices.
    rows = np.broadcast_to(np.zeros((), dtype=np.float32), (1000, 1536))
    captured.layers = dict.fromkeys(captured.layer_ids, rows)
    sentinel = StatisticalControlVector(method="pca", directions={}, metadata={})
    calls = []

    def estimator(method, pooled, positive, negative, **options):
        assert method == "pca" and options["variant"] == "center"
        assert options["normalize"] is normalize
        assert pooled.capture is captured
        assert positive == list(range(500))
        assert negative == list(range(500, 1000))
        calls.append(method)
        return sentinel

    monkeypatch.setattr(
        "easysteer.extraction.api.extract_statistical_control_vector", estimator
    )
    # Confirm the workload needs the notebook's explicit budget override.
    with pytest.raises(MemoryError, match="max_working_bytes"):
        extract(captured, [True] * 500 + [False] * 500, method="pca", variant="center")
    path = f"replications/{notebook}/{notebook}.ipynb"
    [code] = _cells_calling(path, "extract")
    namespace = {"result": captured, count_name: list(range(500))}
    exec(code, namespace)  # noqa: S102 - Execute repository-owned notebook code.
    assert calls == ["pca"]
    assert exports["MATH500.gguf"] is sentinel


@pytest.mark.parametrize("workflow", ["math", "seal"])
def test_paragraph_averages_preserve_global_sample_mapping(workflow, exports):
    categories = [
        {0: "Execution", 2: "Reflection", 5: "Execution"},
        {},
        {3: "Transition"},
        {1: "Execution", 2: "Transition", 4: "Transition", 6: "Reflection"},
    ]
    traces = [[i] * 8 for i in range(len(categories))]
    layers = (10, 20)

    def batches():
        active = [0, 2, 3] if workflow == "math" else [0, 1, 2, 3]
        # The byte budget can split before the configured prompt-count boundary.
        for indices in ([0], list(range(1, len(active)))):
            originals = [active[i] for i in indices]
            yield _Capture(
                {
                    layer: [
                        np.array([[i, pos, layer] for pos in categories[i]]).reshape(
                            -1, 3
                        )
                        for i in originals
                    ]
                    for layer in layers
                },
                sample_indices=indices,
                positions=[list(categories[i]) for i in originals],
            )

    namespace = {"batches": batches(), "category_by_position": categories}
    if workflow == "math":

        def capture_batches(llm, prompts, *, per_prompt_selects, **kwargs):
            active = [0, 2, 3]
            assert list(prompts) == [{"prompt_token_ids": traces[i]} for i in active]
            assert [spec.prompt_positions for spec in per_prompt_selects] == [
                list(categories[i]) for i in active
            ]
            return batches()

        namespace.update(
            np=np,
            capture_batches=capture_batches,
            SelectSpec=SimpleNamespace,
            trace_ids=traces,
            llm=None,
            CAPTURE_BATCH_SIZE=32,
            MODEL="fixture",
            release_capture_cache=lambda llm: None,
            StatisticalControlVector=StatisticalControlVector,
        )
        path = "experiment/math/data_construction.ipynb"
    else:
        path = "replications/seal/seal.ipynb"
    [code] = _cells_calling(path, "StatisticalControlVector")
    exec(code, namespace)  # noqa: S102 - Execute repository-owned notebook code.

    for category in ("Execution", "Reflection", "Transition"):
        vector = exports[f"{category.lower()}_avg_vector.gguf"]
        for layer in layers:
            rows = [
                [i, pos, layer]
                for i, mapping in enumerate(categories)
                for pos, label in mapping.items()
                if label == category
            ]
            np.testing.assert_allclose(vector.directions[layer], np.mean(rows, axis=0))
        assert vector.metadata["num_vectors_averaged"] == len(rows)
        assert vector.component == "hidden_states"
        assert vector.model_type == "fixture"


def test_sake_uses_sample_order_for_source_and_target():
    samples = np.random.default_rng(42).normal(size=(8, 1, 3)).astype(np.float32)
    namespace = {
        "result": _Capture({31: list(samples)}),
        "source_prompts": list(range(4)),
    }
    [code] = _cells_calling("replications/sake/sake.ipynb", "_psd_sqrtm")
    exec(code, namespace)  # noqa: S102 - Execute repository-owned notebook code.
    np.testing.assert_array_equal(namespace["Xs"], samples[:4, 0])
    np.testing.assert_array_equal(namespace["Xt"], samples[4:, 0])
    np.testing.assert_allclose(
        namespace["A"] @ namespace["mu_s"] + namespace["b"], namespace["mu_t"]
    )
