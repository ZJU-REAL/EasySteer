# SPDX-License-Identifier: Apache-2.0
"""Bounded labelled extraction and target-preserving serialization."""

import gc
import weakref

import numpy as np
import pytest

from easysteer.extraction import StatisticalControlVector, extract


class ArrayCapture:
    """The CaptureResult row protocol, without a torch/runtime dependency."""

    def __init__(
        self, samples, *, component="hidden_states", model="tiny", selection=None
    ):
        self.samples = samples
        self.layer_ids = sorted(samples[0]) if samples else []
        self.layers = {
            layer: np.concatenate([sample[layer] for sample in samples])
            for layer in self.layer_ids
        }
        self.component = component
        self.model = model
        self.selection = selection

    def __len__(self):
        return len(self.samples)

    def sample_rows(self, sample, layer):
        return self.samples[sample][layer]

    def iter_sample_rows(self, sample, layer, *, chunk_size=32):
        rows = self.samples[sample][layer]
        for start in range(0, len(rows), chunk_size):
            yield rows[start : start + chunk_size]

    def token(self, sample, layer, position=-1):
        return self.samples[sample][layer][position]


def capture_rows(rows, **kwargs):
    return ArrayCapture([{3: np.atleast_2d(row)} for row in rows], **kwargs)


def test_diffmean_weights_samples_equally_and_preserves_true_layers():
    samples = [
        {3: np.array([[1.0, 4.0], [3.0, 6.0]]), 9: np.array([[3.0], [5.0]])},
        {3: np.array([[8.0, 1.0]]), 9: np.array([[10.0]])},
        {3: np.tile([1.0, 2.0], (7, 1)), 9: np.ones((7, 1))},
    ]
    streamed = extract(
        (ArrayCapture(samples[i : i + 1]) for i in range(3)),
        iter([1, True, 0]),
        token_pos="mean",
        normalize=False,
    )
    np.testing.assert_allclose(streamed.directions[3], [4.0, 1.0])
    np.testing.assert_allclose(streamed.directions[9], [6.0])
    assert streamed.component == "hidden_states"
    assert streamed.model_type == "tiny"
    assert streamed.metadata["n_positive"] == 2
    assert streamed.metadata["n_negative"] == 1


def test_consumed_capture_batches_are_released_before_next_capture():
    def batches():
        for value in range(20):
            batch = capture_rows([[value, 1.0]])
            reference = weakref.ref(batch)
            yield batch
            del batch
            gc.collect()
            assert reference() is None

    result = extract(batches(), [i % 2 for i in range(20)], normalize=False)
    np.testing.assert_allclose(result.directions[3], [1.0, 0.0])


@pytest.mark.parametrize(
    "labels, message",
    [
        ([1], "ended before"),
        ([1, 0, 1], "more entries"),
        ([1, 0.0], "Boolean or integer"),
        ([1, 2], "Boolean or integer"),
        ([1, 1], "negative samples"),
    ],
)
def test_invalid_labels_fail_explicitly(labels, message):
    with pytest.raises(ValueError, match=message):
        extract(capture_rows([[1.0], [2.0]]), labels)


@pytest.mark.parametrize(
    "changed",
    [
        {"component": "router_logits"},
        {"model": "other"},
        {"selection": {"prompt_positions": [-1]}},
    ],
)
def test_capture_provenance_must_agree_across_batches(changed):
    with pytest.raises(ValueError, match="identical layers"):
        extract([capture_rows([[1.0]]), capture_rows([[2.0]], **changed)], [1, 0])


def test_budget_rejects_before_reducing_rows():
    batch = capture_rows([[1.0, 2.0], [3.0, 4.0]])
    batch.token = lambda *args: pytest.fail("allocation guard must precede row reads")
    with pytest.raises(MemoryError, match="max_working_bytes"):
        extract(batch, [1, 0], max_working_bytes=1)
    with pytest.raises(MemoryError, match="max_working_bytes"):
        extract(batch, [1, 0], method="linear_probe", max_working_bytes=1)


@pytest.mark.parametrize("budget", [None, 0, -1, True, 1.5])
def test_budget_requires_a_finite_positive_integer(budget):
    with pytest.raises(ValueError, match="max_working_bytes"):
        extract([], [], max_working_bytes=budget)


@pytest.mark.parametrize("token_pos", [-2, "mean"])
def test_missing_selected_rows_fail_with_sample_context(token_pos):
    batch = ArrayCapture([{3: np.empty((0, 2))}])
    with pytest.raises(ValueError, match="sample 0, layer 3"):
        extract(batch, [1], token_pos=token_pos)


def test_stream_pca_is_explicit_and_does_not_consume_unsupported_input():
    def batches():
        pytest.fail("unsupported method must fail before consuming captures")
        yield

    with pytest.raises(ValueError, match="single materialized"):
        extract(batches(), [], method="pca")


def test_incremental_pca_fits_rank_one_data_and_orients_with_class_means():
    direction = np.array([0.6, 0.8])
    rows = np.arange(1.0, 18.0)[:, None] * direction
    batches = (capture_rows(rows[i : i + 3]) for i in range(0, len(rows), 3))
    result = extract(
        batches,
        [0] * 2 + [1] * 15,
        method="incremental_pca",
        pca_batch_size=4,
    )
    np.testing.assert_allclose(result.directions[3], direction, atol=1e-6)
    assert result.metadata["approximate"] is True
    assert result.metadata["explained_variance"][3] == pytest.approx(1.0)


def test_incremental_pca_requires_two_positive_rows():
    with pytest.raises(ValueError, match="at least two"):
        extract(capture_rows([[1.0], [2.0]]), [0, 1], method="incremental_pca")


def test_single_capture_delegates_pca_variant_without_method_collision():
    rows = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    result = extract(capture_rows(rows), [1, 1, 0, 0], method="pca", variant="center")
    assert result.method == "pca_center"
    assert result.component == "hidden_states"


def test_gguf_roundtrip_preserves_metadata_types_and_target(tmp_path):
    pytest.importorskip("gguf")
    metadata = {
        "normalize": False,
        "token_pos": "mean",
        "n_positive": 123,
        "selected_heads": {3: [1, 2], "3": [7]},
        "window": (-3, None),
        "nested": {"note": "capture ☃", "values": [1.25, None, True]},
    }
    vector = StatisticalControlVector(
        "iti",
        {3: np.array([1.0, 2.0], dtype=np.float32)},
        metadata,
        model_type="some/model",
        component="attention_heads",
    )
    path = tmp_path / "vector.gguf"
    vector.export_gguf(path)
    restored = StatisticalControlVector.import_gguf(path)
    assert restored.metadata == metadata
    assert restored.component == "attention_heads"
    assert restored.model_type == "some/model"
    np.testing.assert_array_equal(restored.directions[3], vector.directions[3])


def test_legacy_gguf_reads_value_types_including_strings(tmp_path):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "legacy.gguf"
    writer = gguf.GGUFWriter(path, "controlvector")
    writer.add_string("controlvector.model_hint", "legacy")
    writer.add_string("controlvector.method", "pca")
    writer.add_string("controlvector.token_pos", "last")
    writer.add_float32("controlvector.variance", 0.5)
    writer.add_uint32("controlvector.n_positive", 7)
    writer.add_bool("controlvector.normalize", True)
    writer.add_tensor("direction.8", np.array([1.0], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    restored = StatisticalControlVector.import_gguf(path)
    assert restored.component is None
    assert restored.metadata == {
        "token_pos": "last",
        "variance": 0.5,
        "n_positive": 7,
        "normalize": True,
    }


@pytest.mark.parametrize(
    "component, algorithm",
    [
        ("hidden_states", "direct"),
        ("attention_heads", "attention_add"),
    ],
)
def test_vector_spec_preserves_target_and_requires_explicit_apply(component, algorithm):
    pytest.importorskip("torch")
    from vllm.steer_vectors import ApplySpec

    vector = StatisticalControlVector("diffmean", {3: np.ones(2)}, component=component)
    apply = ApplySpec(generation_window=(0, 3))
    spec = vector.to_spec(apply=apply, scale=2.0)
    assert spec.vectors[0].algorithm == algorithm
    assert spec.vectors[0].apply == apply
    assert spec.vectors[0].layers == [3]
    assert spec.vectors[0].scale == 2.0
    with pytest.raises(TypeError, match="apply"):
        vector.to_spec()


@pytest.mark.parametrize("component", [None, "router_logits"])
def test_vector_spec_does_not_guess_an_unknown_or_router_target(component):
    vector = StatisticalControlVector("diffmean", {3: np.ones(2)}, component=component)
    with pytest.raises(ValueError, match="Cannot convert"):
        vector.to_spec(apply={"prompt": "all"})


def test_pooling_reads_bounded_chunks_without_materializing_a_sample():
    rows = np.arange(400.0).reshape(100, 4)
    batch = ArrayCapture([{3: rows}, {3: np.zeros((1, 4))}])
    batch.sample_rows = lambda *args: pytest.fail("pooling must use bounded row chunks")
    result = extract(batch, [1, 0], token_pos="mean", normalize=False)
    np.testing.assert_allclose(result.directions[3], rows.mean(0))


def test_capture_chunk_reader_bounds_interleaved_gathers():
    torch = pytest.importorskip("torch")
    from easysteer.capture import CaptureResult

    # The reader only needs the indexed rows; construction/label matching is
    # exercised by the CaptureResult integration tests.
    capture = object.__new__(CaptureResult)
    capture.layers = {3: torch.arange(80).reshape(20, 4)}
    capture._sample_rows = [list(range(0, 20, 2))]
    chunks = list(capture.iter_sample_rows(0, 3, chunk_size=3))
    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 1]
    assert torch.equal(torch.cat(chunks), capture.layers[3][::2])


@pytest.mark.parametrize("method", ["pca", "lat", "linear_probe"])
def test_materialized_estimators_use_bounded_token_pooling(method):
    rng = np.random.default_rng(4)
    samples = [{3: rng.normal(loc=i, size=(65, 4))} for i in range(8)]
    selection = {"prompt": "all"}
    capture = ArrayCapture(samples, selection=selection)
    capture.sample_rows = lambda *args: pytest.fail("must not gather the full sample")
    result = extract(capture, [0] * 4 + [1] * 4, method=method, token_pos="mean")
    assert result.metadata["token_pos"] == "mean"
    assert result.metadata["capture_selection"] == selection


def test_incremental_pca_releases_the_full_svd_matrix(monkeypatch):
    from scipy import linalg

    from easysteer.extraction.streaming import _IncrementalLayer

    original = linalg.svd
    bases = []

    def svd(*args, **kwargs):
        result = original(*args, **kwargs)
        bases.append(weakref.ref(result[2]))
        return result

    monkeypatch.setattr(linalg, "svd", svd)
    layer = _IncrementalLayer(64, 4)
    for row in np.random.default_rng(6).normal(size=(8, 64)):
        layer.update(row)
    gc.collect()
    assert all(reference() is None for reference in bases)
    assert layer.pca.components_.shape == (1, 64)


def test_single_capture_does_not_allocate_more_pca_rows_than_samples():
    rows = np.arange(4.0)[:, None] * np.array([0.6, 0.8])
    result = extract(
        capture_rows(rows),
        [1] * 4,
        method="incremental_pca",
        pca_batch_size=100_000,
        max_working_bytes=20_000,
    )
    np.testing.assert_allclose(np.abs(result.directions[3]), [0.6, 0.8], atol=1e-6)
    assert result.metadata["pca_batch_size"] == 100_000


@pytest.mark.parametrize("variant", ["standard", "diff", "center"])
@pytest.mark.parametrize("normalize", [False, True])
def test_materialized_pca_matches_the_existing_extractor(variant, normalize):
    from easysteer.extraction import PCAExtractor

    rows = np.random.default_rng(12).normal(size=(12, 7))
    capture = capture_rows(rows)
    positive = [0, 2, 4, 6, 8, 10]
    negative = [1, 3, 5, 7, 9, 11]
    expected = PCAExtractor.extract(
        capture,
        positive,
        negative,
        variant=variant,
        normalize=normalize,
    )
    result = extract(
        capture,
        [i % 2 == 0 for i in range(12)],
        method="pca",
        variant=variant,
        normalize=normalize,
    )
    np.testing.assert_allclose(result.directions[3], expected.directions[3], atol=1e-6)


def test_float64_capture_precision_and_scalar_tensor_labels_are_preserved():
    torch = pytest.importorskip("torch")
    from easysteer.extraction.streaming import _to_numpy

    positive = torch.tensor([1.0 + 2**-40, 2.0], dtype=torch.float64)
    negative = torch.tensor([1.0, 1.0], dtype=torch.float64)
    assert _to_numpy(positive).dtype == np.float64
    capture = capture_rows([_to_numpy(positive), _to_numpy(negative)])
    result = extract(capture, torch.tensor([1, 0]), normalize=False)
    np.testing.assert_array_equal(
        result.directions[3], np.array([2**-40, 1.0], dtype=np.float32)
    )


def test_iti_validation_width_mismatch_fails_before_feature_reads():
    training = capture_rows([[1.0, 2.0], [3.0, 4.0]], component="attention_heads")
    validation = capture_rows(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], component="attention_heads"
    )
    training.token = lambda *args: pytest.fail(
        "width validation must precede row reads"
    )
    with pytest.raises(ValueError, match="same layer widths"):
        extract(
            training,
            [1, 0],
            method="iti",
            validation_hidden_states=validation,
            validation_positive_indices=[0],
        )


def test_iti_accepts_labelled_validation_capture_and_bounded_mean_pooling():
    positive = np.tile([2.0, 0.0], (65, 1))
    negative = -positive
    samples = [{3: negative}, {3: negative * 2}, {3: positive}, {3: positive * 2}]
    training = ArrayCapture(samples, component="attention_heads")
    validation = ArrayCapture(samples, component="attention_heads")
    for capture in (training, validation):
        capture.layouts = {3: {"width": 2, "num_heads": 1, "head_size": 2}}
        capture.sample_rows = lambda *args: pytest.fail("ITI must use bounded pooling")
    result = extract(
        training,
        [0, 0, 1, 1],
        method="iti",
        token_pos="mean",
        validation_hidden_states=validation,
        validation_labels=iter([0, 0, 1, 1]),
    )
    assert result.component == "attention_heads"
    assert result.metadata["normalize"] is False
    assert result.directions[3][0] > 0
    assert result.directions[3][1] == 0


def test_selection_override_provenance_does_not_retain_sample_lists():
    first = capture_rows([[1.0]])
    second = capture_rows([[2.0]])
    second.per_prompt_selections = [{"prompt_positions": [-2]}]
    result = extract([first, second], [0, 1])
    assert result.metadata["capture_has_per_prompt_selections"] is True
    assert result.metadata["capture_selection"] is None
    assert "per_prompt_selections" not in result.metadata
