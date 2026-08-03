# SPDX-License-Identifier: Apache-2.0
"""CPU units for the feature-extractor layer (easysteer.steer).

Pins the explicit-failure conventions: one shared negative-derivation
rule (negatives are every sample not in positive_indices, ascending;
positive_indices never rebound), loud rejection of unknown options,
regularizations and PCA methods, and layer dicts keyed by TRUE layer
ids flowing through utils and the extractors.
"""

import numpy as np
import pytest

from easysteer.steer import (
    DiffMeanExtractor,
    LATExtractor,
    LinearProbeExtractor,
    PCAExtractor,
    derive_negative_indices,
    extract_diffmean_control_vector,
    extract_statistical_control_vector,
    extract_token_hiddens,
)

RNG = np.random.default_rng(11)
DIM = 8
N_SAMPLES = 6
N_TOKENS = 3


def make_nested(n_samples=N_SAMPLES, layer_count=2, offset_indices=()):
    """Nested [sample][layer][token] data; offset samples separable."""
    nested = []
    for i in range(n_samples):
        shift = 2.0 if i in offset_indices else 0.0
        nested.append(
            [
                [RNG.normal(size=DIM) + shift for _ in range(N_TOKENS)]
                for _ in range(layer_count)
            ]
        )
    return nested


class FakeCapture:
    """Duck-typed CaptureResult: true layer ids + nested conversion."""

    def __init__(self, nested, layer_ids):
        self._nested = nested
        self._layer_ids = list(layer_ids)

    @property
    def layer_ids(self):
        return list(self._layer_ids)

    def to_nested(self):
        return self._nested

    def __len__(self):
        return len(self._nested)


class TestNegativeDerivationRule:
    def test_complement_ascending(self):
        assert derive_negative_indices(6, [1, 3]) == [0, 2, 4, 5]
        assert derive_negative_indices(4, []) == [0, 1, 2, 3]
        assert derive_negative_indices(3, [0, 1, 2]) == []

    def test_positive_indices_never_discarded(self):
        # Positives at the END of the batch: the old first-half
        # convention would silently replace them with [0, 1, 2].
        nested = make_nested(offset_indices=(4, 5))
        pos_h, neg_h = extract_token_hiddens(nested, [4, 5])
        assert pos_h[0].shape[0] == 2
        assert neg_h[0].shape[0] == 4
        expected = np.vstack([nested[4][0][-1], nested[5][0][-1]])
        assert np.allclose(pos_h[0], expected)

    def test_derived_equals_explicit_complement_diffmean(self):
        nested = make_nested(offset_indices=(1, 3, 5))
        derived = DiffMeanExtractor.extract(nested, [1, 3, 5])
        explicit = DiffMeanExtractor.extract(
            nested, [1, 3, 5], negative_indices=[0, 2, 4]
        )
        for layer in derived.directions:
            assert np.allclose(
                derived.directions[layer], explicit.directions[layer]
            )
        assert derived.metadata["n_negative"] == 3

    def test_derived_equals_explicit_complement_pca_diff(self):
        nested = make_nested(offset_indices=(1, 3, 5))
        derived = PCAExtractor.extract(nested, [1, 3, 5], method="diff")
        explicit = PCAExtractor.extract(
            nested, [1, 3, 5], negative_indices=[0, 2, 4], method="diff"
        )
        for layer in derived.directions:
            assert np.allclose(
                derived.directions[layer], explicit.directions[layer]
            )

    def test_derived_equals_explicit_complement_linear_probe(self):
        nested = make_nested(offset_indices=(1, 3, 5))
        derived = LinearProbeExtractor.extract(nested, [1, 3, 5])
        explicit = LinearProbeExtractor.extract(
            nested, [1, 3, 5], negative_indices=[0, 2, 4]
        )
        for layer in derived.directions:
            assert np.allclose(
                derived.directions[layer], explicit.directions[layer]
            )

    def test_derived_equals_explicit_complement_lat(self):
        nested = make_nested(offset_indices=(1, 3, 5))
        np.random.seed(0)
        derived = LATExtractor.extract(
            nested, [1, 3, 5], use_positive_only=False
        )
        np.random.seed(0)
        explicit = LATExtractor.extract(
            nested, [1, 3, 5], negative_indices=[0, 2, 4],
            use_positive_only=False,
        )
        for layer in derived.directions:
            assert np.allclose(
                derived.directions[layer], explicit.directions[layer]
            )


class TestExplicitOptionValidation:
    def test_unknown_regularization_raises(self):
        nested = make_nested(offset_indices=(0, 1, 2))
        with pytest.raises(ValueError, match="ridge"):
            LinearProbeExtractor.extract(
                nested, [0, 1, 2], regularization="ridge"
            )
        with pytest.raises(ValueError, match="elasticnet"):
            LinearProbeExtractor.extract(
                nested, [0, 1, 2], regularization="ridge"
            )

    def test_effective_penalty_recorded(self):
        nested = make_nested(offset_indices=(0, 1, 2))
        vec = LinearProbeExtractor.extract(
            nested, [0, 1, 2], regularization="none"
        )
        assert vec.metadata["regularization"] == "none"

    def test_unknown_pca_method_raises(self):
        nested = make_nested(offset_indices=(0, 1, 2))
        with pytest.raises(ValueError, match="Unknown PCA method"):
            PCAExtractor.extract(nested, [0, 1, 2], method="bogus")

    def test_pca_n_components_must_be_one(self):
        nested = make_nested(offset_indices=(0, 1, 2))
        with pytest.raises(ValueError, match="n_components"):
            PCAExtractor.extract(nested, [0, 1, 2], n_components=2)
        vec = PCAExtractor.extract(nested, [0, 1, 2], n_components=1)
        assert vec.metadata["n_components"] == 1

    def test_unknown_kwarg_raises_with_method_name(self):
        nested = make_nested(offset_indices=(0, 1, 2))
        with pytest.raises(ValueError, match="diffmean") as excinfo:
            extract_diffmean_control_vector(nested, [0, 1, 2], normalise=True)
        # The typo and the accepted spelling are both named.
        assert "normalise" in str(excinfo.value)
        assert "normalize" in str(excinfo.value)

    def test_unknown_kwarg_rejected_per_method(self):
        nested = make_nested(offset_indices=(0, 1, 2))
        # use_positive_only is a LAT option, not a PCA one.
        with pytest.raises(ValueError, match="pca"):
            extract_statistical_control_vector(
                "pca", nested, [0, 1, 2], use_positive_only=True
            )
        extract_statistical_control_vector(
            "lat", nested, [0, 1, 2, 3], use_positive_only=True
        )

    def test_unknown_method_raises(self):
        nested = make_nested(offset_indices=(0, 1, 2))
        with pytest.raises(ValueError, match="Unsupported method"):
            extract_statistical_control_vector("mystery", nested, [0, 1, 2])


class TestTrueLayerIdKeys:
    def test_extract_token_hiddens_keys_true_ids(self):
        capture = FakeCapture(make_nested(offset_indices=(0, 1, 2)), [10, 20])
        pos_h, neg_h = extract_token_hiddens(capture, [0, 1, 2])
        assert sorted(pos_h) == [10, 20]
        assert sorted(neg_h) == [10, 20]
        assert pos_h[10].shape == (3, DIM)
        assert neg_h[20].shape == (3, DIM)

    def test_linear_probe_keys_true_ids(self):
        capture = FakeCapture(make_nested(offset_indices=(0, 1, 2)), [10, 20])
        vec = LinearProbeExtractor.extract(capture, [0, 1, 2])
        assert sorted(vec.directions) == [10, 20]
        assert sorted(vec.metadata["classification_scores"]) == [10, 20]

    def test_diffmean_keys_true_ids(self):
        nested = make_nested(offset_indices=(0, 1, 2))
        capture = FakeCapture(nested, [10, 20])
        vec = DiffMeanExtractor.extract(capture, [0, 1, 2])
        assert sorted(vec.directions) == [10, 20]
        plain = DiffMeanExtractor.extract(nested, [0, 1, 2])
        assert np.allclose(vec.directions[20], plain.directions[1])
