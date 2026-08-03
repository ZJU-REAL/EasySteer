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
    MomentsAccumulator,
    PCAExtractor,
    derive_negative_indices,
    extract_diffmean_control_vector,
    extract_statistical_control_vector,
    extract_token_hiddens,
)
from easysteer.steer.utils import (
    _TOKEN_REDUCERS,
    correct_sign,
    extract_token_from_sequence,
    l2_normalize,
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


UNIFIED_METADATA_KEYS = {"normalize", "n_positive", "n_negative"}
OLD_METADATA_KEYS = {"normalized", "num_positive", "num_negative"}


class TestSharedHelpers:
    def test_l2_normalize_unit_norm(self):
        out = l2_normalize(np.array([3.0, 4.0]))
        assert np.allclose(out, [0.6, 0.8])
        assert np.isclose(np.linalg.norm(out), 1.0)

    def test_l2_normalize_zero_vector_unchanged(self):
        v = np.zeros(4)
        assert np.allclose(l2_normalize(v), v)

    def test_correct_sign_flips_when_means_invert(self):
        component = np.array([1.0, 0.0])
        pos = np.array([[2.0, 0.0], [3.0, 0.0]])
        neg = np.array([[-2.0, 0.0], [-3.0, 0.0]])
        # Already points from negatives toward positives: unchanged.
        assert np.allclose(correct_sign(component, pos, neg), component)
        # Invert the roles of the two means: the sign must flip.
        assert np.allclose(correct_sign(component, neg, pos), -component)

    def test_correct_sign_near_zero_vector_unchanged(self):
        zero = np.zeros(2)
        pos = np.ones((2, 2))
        neg = -np.ones((2, 2))
        assert np.allclose(correct_sign(zero, pos, neg), zero)


class TestTokenReducers:
    SEQ = [np.array([1.0, 0.0]), np.array([0.0, 3.0]), np.array([2.0, 2.0])]

    def test_reducer_table_names(self):
        assert sorted(_TOKEN_REDUCERS) == [
            "first", "last", "max", "mean", "min",
        ]

    def test_int_and_positional_names(self):
        assert np.allclose(extract_token_from_sequence(self.SEQ, 1), [0.0, 3.0])
        assert np.allclose(
            extract_token_from_sequence(self.SEQ, -1), [2.0, 2.0]
        )
        assert np.allclose(
            extract_token_from_sequence(self.SEQ, "first"), [1.0, 0.0]
        )
        assert np.allclose(
            extract_token_from_sequence(self.SEQ, "last"), [2.0, 2.0]
        )

    def test_mean_max_min(self):
        assert np.allclose(
            extract_token_from_sequence(self.SEQ, "mean"), [1.0, 5.0 / 3.0]
        )
        # L2 norms are [1, 3, sqrt(8)]: max picks [0, 3], min [1, 0].
        assert np.allclose(
            extract_token_from_sequence(self.SEQ, "max"), [0.0, 3.0]
        )
        assert np.allclose(
            extract_token_from_sequence(self.SEQ, "min"), [1.0, 0.0]
        )

    def test_unsupported_position_raises(self):
        with pytest.raises(ValueError, match="token_pos"):
            extract_token_from_sequence(self.SEQ, "median")


class TestUnifiedMetadata:
    def test_extract_paths_use_unified_keys(self):
        nested = make_nested(offset_indices=(1, 3, 5))
        vectors = [
            DiffMeanExtractor.extract(nested, [1, 3, 5]),
            PCAExtractor.extract(nested, [1, 3, 5], method="diff"),
            LinearProbeExtractor.extract(nested, [1, 3, 5]),
        ]
        np.random.seed(0)
        vectors.append(
            LATExtractor.extract(nested, [1, 3, 5], use_positive_only=False)
        )
        for vec in vectors:
            keys = set(vec.metadata)
            assert UNIFIED_METADATA_KEYS <= keys
            assert not (OLD_METADATA_KEYS & keys)
            assert vec.metadata["n_positive"] == 3
            assert vec.metadata["n_negative"] == 3

    def test_from_moments_shares_the_extract_vocabulary(self):
        pos_rows = RNG.normal(size=(6, DIM)) + 1.0
        neg_rows = RNG.normal(size=(6, DIM))
        pos_acc, neg_acc = MomentsAccumulator(), MomentsAccumulator()
        pos_acc.update(0, pos_rows)
        neg_acc.update(0, neg_rows)
        streamed = DiffMeanExtractor.from_moments(pos_acc, neg_acc)

        nested = make_nested(offset_indices=(1, 3, 5))
        batch = DiffMeanExtractor.extract(nested, [1, 3, 5])

        streamed_keys = set(streamed.metadata)
        batch_keys = set(batch.metadata)
        # Both paths agree on one count/normalize vocabulary ...
        assert (
            streamed_keys & (UNIFIED_METADATA_KEYS | OLD_METADATA_KEYS)
            == batch_keys & (UNIFIED_METADATA_KEYS | OLD_METADATA_KEYS)
            == UNIFIED_METADATA_KEYS
        )
        assert streamed.metadata["n_positive"] == 6
        assert streamed.metadata["n_negative"] == 6

    def test_pca_from_moments_uses_unified_keys(self):
        rows = RNG.normal(size=(8, DIM))
        moments = MomentsAccumulator(track_second_moment=True)
        moments.update(0, rows)
        vec = PCAExtractor.from_moments(moments)
        keys = set(vec.metadata)
        assert UNIFIED_METADATA_KEYS <= keys
        assert not (OLD_METADATA_KEYS & keys)
        assert vec.metadata["n_positive"] == 8
        assert vec.metadata["n_negative"] == 0


class TestLATSingleExtraction:
    def test_extract_token_hiddens_called_once(self, monkeypatch):
        """LAT with direction correction must tokenize the corpus once.

        The pre-refactor implementation re-ran extract_token_hiddens
        inside its per-layer loop for the correction step.
        """
        import easysteer.steer.base_extractor as base_extractor

        calls = {"n": 0}
        real = base_extractor.extract_token_hiddens

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(base_extractor, "extract_token_hiddens", counting)
        nested = make_nested(offset_indices=(1, 3, 5))
        np.random.seed(0)
        vec = LATExtractor.extract(
            nested, [1, 3, 5], use_positive_only=False, correct_direction=True
        )
        assert calls["n"] == 1
        assert sorted(vec.directions) == [0, 1]
