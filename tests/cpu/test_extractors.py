# SPDX-License-Identifier: Apache-2.0
"""CPU units for the feature-extractor layer (easysteer.extraction).

Pins the explicit-failure conventions: one shared negative-derivation
rule (negatives are every sample not in positive_indices, ascending;
positive_indices never rebound), loud rejection of unknown options,
regularizations and PCA methods, and layer dicts keyed by TRUE layer
ids flowing through utils and the extractors.
"""

import numpy as np
import pytest

from easysteer.extraction import (
    DiffMeanExtractor,
    ITIExtractor,
    LATExtractor,
    LinearProbeExtractor,
    MomentsAccumulator,
    PCAExtractor,
    derive_negative_indices,
    extract_diffmean_control_vector,
    extract_statistical_control_vector,
    extract_token_hiddens,
)
from easysteer.extraction._utils import correct_sign, l2_normalize
from easysteer.extraction.selection import (
    _TOKEN_REDUCERS,
    extract_token_from_sequence,
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
    """CaptureResult's row-reading contract without requiring torch."""

    def __init__(self, nested, layer_ids):
        self._nested = nested
        self._layer_ids = list(layer_ids)

    @property
    def layer_ids(self):
        return list(self._layer_ids)

    def to_nested(self):
        raise AssertionError("extractors must not materialize the full nested capture")

    def sample_rows(self, sample, layer):
        return self._nested[sample][self._layer_ids.index(layer)]

    def token(self, sample, layer, position=-1):
        return self.sample_rows(sample, layer)[position]

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
            assert np.allclose(derived.directions[layer], explicit.directions[layer])
        assert derived.metadata["n_negative"] == 3

    def test_derived_equals_explicit_complement_pca_diff(self):
        nested = make_nested(offset_indices=(1, 3, 5))
        derived = PCAExtractor.extract(nested, [1, 3, 5], method="diff")
        explicit = PCAExtractor.extract(
            nested, [1, 3, 5], negative_indices=[0, 2, 4], method="diff"
        )
        for layer in derived.directions:
            assert np.allclose(derived.directions[layer], explicit.directions[layer])

    def test_derived_equals_explicit_complement_linear_probe(self):
        nested = make_nested(offset_indices=(1, 3, 5))
        derived = LinearProbeExtractor.extract(nested, [1, 3, 5])
        explicit = LinearProbeExtractor.extract(
            nested, [1, 3, 5], negative_indices=[0, 2, 4]
        )
        for layer in derived.directions:
            assert np.allclose(derived.directions[layer], explicit.directions[layer])

    def test_derived_equals_explicit_complement_lat(self):
        nested = make_nested(offset_indices=(1, 3, 5))
        np.random.seed(0)
        derived = LATExtractor.extract(nested, [1, 3, 5], use_positive_only=False)
        np.random.seed(0)
        explicit = LATExtractor.extract(
            nested,
            [1, 3, 5],
            negative_indices=[0, 2, 4],
            use_positive_only=False,
        )
        for layer in derived.directions:
            assert np.allclose(derived.directions[layer], explicit.directions[layer])


class TestExplicitOptionValidation:
    def test_unknown_regularization_raises(self):
        nested = make_nested(offset_indices=(0, 1, 2))
        with pytest.raises(ValueError, match="ridge") as error:
            LinearProbeExtractor.extract(nested, [0, 1, 2], regularization="ridge")
        assert "elasticnet" in str(error.value)

    def test_effective_penalty_recorded(self):
        nested = make_nested(offset_indices=(0, 1, 2))
        vec = LinearProbeExtractor.extract(nested, [0, 1, 2], regularization="none")
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
            "first",
            "last",
            "max",
            "mean",
            "min",
        ]

    def test_int_and_positional_names(self):
        assert np.allclose(extract_token_from_sequence(self.SEQ, 1), [0.0, 3.0])
        assert np.allclose(extract_token_from_sequence(self.SEQ, -1), [2.0, 2.0])
        assert np.allclose(extract_token_from_sequence(self.SEQ, "first"), [1.0, 0.0])
        assert np.allclose(extract_token_from_sequence(self.SEQ, "last"), [2.0, 2.0])

    def test_mean_max_min(self):
        assert np.allclose(
            extract_token_from_sequence(self.SEQ, "mean"), [1.0, 5.0 / 3.0]
        )
        # L2 norms are [1, 3, sqrt(8)]: max picks [0, 3], min [1, 0].
        assert np.allclose(extract_token_from_sequence(self.SEQ, "max"), [0.0, 3.0])
        assert np.allclose(extract_token_from_sequence(self.SEQ, "min"), [1.0, 0.0])

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
        vectors.append(LATExtractor.extract(nested, [1, 3, 5], use_positive_only=False))
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
    def test_capture_rows_are_visited_once(self, monkeypatch):
        """LAT reuses selected rows for direction correction within each layer."""
        from easysteer.extraction import base as base_extractor

        calls = {"n": 0}
        real = base_extractor.iter_token_hiddens

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(base_extractor, "iter_token_hiddens", counting)
        nested = make_nested(offset_indices=(1, 3, 5))
        np.random.seed(0)
        vec = LATExtractor.extract(
            nested, [1, 3, 5], use_positive_only=False, correct_direction=True
        )
        assert calls["n"] == 1
        assert sorted(vec.directions) == [0, 1]


class TestITIExtraction:
    @staticmethod
    def captures():
        # Head 0 predicts training labels but reverses on validation.
        train = np.array(
            [
                [4, 0, 3, 0],
                [2, 0, 1, 0],
                [-4, 0, -3, 0],
                [-2, 0, -1, 0],
            ]
        )
        validation = np.array(
            [
                [-4, 0, 5, 0],
                [-2, 0, 3, 0],
                [4, 0, -5, 0],
                [2, 0, -3, 0],
            ]
        )
        return tuple(
            FakeCapture([[[row]] for row in rows], [10]) for rows in (train, validation)
        )

    def test_validation_ranking_development_scale_and_head_layout(self):
        train, validation = self.captures()
        vector = extract_statistical_control_vector(
            "iti",
            train,
            [0, 1],
            validation_hidden_states=validation,
            validation_positive_indices=[0, 1],
            num_heads={10: 2},
        )
        assert list(vector.directions) == [10]
        # std([3, 1, -3, -1, 5, 3, -5, -3]) = sqrt(11).
        np.testing.assert_allclose(vector.directions[10], [0, 0, np.sqrt(11), 0])
        assert vector.metadata["selected_head_scores"] == {"10.1": 1.0}
        assert vector.metadata["normalize"] is False
        assert vector.metadata["n_positive"] == 2
        assert vector.metadata["n_validation_positive"] == 2

    def test_invalid_labels_or_head_layout_rejected(self):
        train, validation = self.captures()
        options = dict(
            validation_hidden_states=validation,
            validation_positive_indices=[0, 1],
        )
        with pytest.raises(ValueError, match="disjoint"):
            ITIExtractor.extract(train, [0, 1], [1, 2], num_heads={10: 2}, **options)
        with pytest.raises(ValueError, match="width"):
            ITIExtractor.extract(train, [0, 1], num_heads={10: 3}, **options)
        with pytest.raises(ValueError, match="top_k"):
            ITIExtractor.extract(train, [0, 1], num_heads={10: 2}, top_k=3, **options)

    def test_degenerate_selected_direction_rejected(self):
        constant = FakeCapture([[[np.ones(4)]]] * 4, [10])
        with pytest.raises(ValueError, match="no nonzero finite ITI direction"):
            ITIExtractor.extract(
                constant,
                [0, 1],
                validation_hidden_states=constant,
                validation_positive_indices=[0, 1],
                num_heads={10: 2},
            )


class TestSampleValidation:
    @pytest.mark.parametrize(
        ("positives", "negatives", "message"),
        [
            ([], [1, 2], "positive_indices must be nonempty"),
            ([0, 0], [1, 2], "unique"),
            ([0, 1], [2, 2], "unique"),
            ([0, 1], [1, 2], "disjoint"),
            ([-1, 0], [1, 2], "integer indices"),
            ([0, 6], [1, 2], "integer indices"),
            ([False, 1], [2, 3], "integer indices"),
            ([0.0, 1], [2, 3], "integer indices"),
        ],
    )
    @pytest.mark.parametrize(
        "extractor",
        [DiffMeanExtractor, PCAExtractor, LATExtractor, LinearProbeExtractor],
    )
    def test_invalid_groups_rejected_before_fitting(
        self, positives, negatives, message, extractor
    ):
        with pytest.raises(ValueError, match=message):
            extractor.extract(make_nested(), positives, negatives)

    @pytest.mark.parametrize(
        ("extractor", "options"),
        [
            (DiffMeanExtractor, {}),
            (PCAExtractor, {"variant": "diff"}),
            (LATExtractor, {"use_positive_only": False}),
            (LinearProbeExtractor, {}),
        ],
    )
    def test_required_negatives_cannot_be_empty(self, extractor, options):
        with pytest.raises(ValueError, match="negative_indices must be nonempty"):
            extractor.extract(make_nested(), list(range(N_SAMPLES)), **options)

    @pytest.mark.parametrize("token_pos", [-1, "mean"])
    def test_missing_capture_rows_explain_sample_layer_and_selection(self, token_pos):
        nested = make_nested()
        nested[1][0] = []
        captured = FakeCapture(nested, [10, 20])
        with pytest.raises(ValueError, match="sample 1, layer 10:.*capture selection"):
            DiffMeanExtractor.extract(captured, [0, 1, 2], token_pos=token_pos)

    def test_empty_capture_layers_are_rejected(self):
        captured = FakeCapture([[], []], [])
        with pytest.raises(ValueError, match="capture layer"):
            DiffMeanExtractor.extract(captured, [0], [1])

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_nonfinite_features_are_rejected(self, value):
        nested = make_nested()
        nested[1][0][-1][0] = value
        with pytest.raises(ValueError, match="sample 1, layer 0:.*finite"):
            DiffMeanExtractor.extract(nested, [0, 1, 2])

    def test_numpy_integer_indices_are_valid(self):
        nested = make_nested()
        vector = DiffMeanExtractor.extract(nested, np.array([0, 1, 2], dtype=np.int64))
        assert len(vector.directions) == 2


class TestClassOptionValidation:
    @pytest.mark.parametrize(
        "extractor",
        [DiffMeanExtractor, PCAExtractor, LATExtractor, LinearProbeExtractor],
    )
    def test_class_entry_points_reject_typos(self, extractor):
        with pytest.raises(TypeError, match="normalise"):
            extractor.extract(make_nested(), [0, 1, 2], normalise=True)


class TestPCAVariants:
    def test_generic_and_specific_compatibility_aliases_agree(self):
        from easysteer.extraction import extract_pca_control_vector

        nested = make_nested(offset_indices=(0, 1, 2))
        calls = [
            lambda: PCAExtractor.extract(nested, [0, 1, 2], variant="diff"),
            lambda: extract_pca_control_vector(nested, [0, 1, 2], method="diff"),
            lambda: extract_statistical_control_vector(
                "pca", nested, [0, 1, 2], method="diff"
            ),
            lambda: extract_statistical_control_vector(
                algorithm="pca",
                all_hidden_states=nested,
                positive_indices=[0, 1, 2],
                variant="diff",
            ),
            lambda: extract_statistical_control_vector(
                method="pca",
                all_hidden_states=nested,
                positive_indices=[0, 1, 2],
                variant="diff",
            ),
        ]
        results = [call() for call in calls]
        for vector in results:
            assert vector.metadata["variant"] == "diff"
            for layer, expected in results[0].directions.items():
                np.testing.assert_allclose(vector.directions[layer], expected)

    def test_conflicting_aliases_are_rejected(self):
        with pytest.raises(ValueError, match="method and variant must agree"):
            PCAExtractor.extract(
                make_nested(), [0, 1, 2], method="diff", variant="center"
            )

    def test_standard_uses_negatives_only_for_sign(self):
        positive = np.array([[-3.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
        negative = np.array([[10.0, 8.0], [10.0, -8.0], [10.0, 0.0]])
        nested = [[[row]] for row in np.vstack([positive, negative])]
        uncorrected = PCAExtractor.extract(nested, [0, 1, 2], correct_direction=False)
        corrected = PCAExtractor.extract(nested, [0, 1, 2], correct_direction=True)
        positive_only = PCAExtractor.extract(nested, [0, 1, 2], [])
        np.testing.assert_allclose(uncorrected.directions[0], [1, 0])
        np.testing.assert_allclose(corrected.directions[0], [-1, 0])
        np.testing.assert_allclose(positive_only.directions[0], [1, 0])
        assert corrected.metadata["n_negative"] == 3


class TestLinearProbeCoordinates:
    @pytest.mark.parametrize("normalize", [False, True])
    def test_standardized_probe_exports_raw_activation_normal(self, normalize):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(71)
        rows = rng.normal(size=(80, 3)) * np.array([100.0, 0.25, 1.0])
        rows[:, 2] = 7.0  # StandardScaler handles constant features with scale 1.
        labels = (rows[:, 0] / 100.0 + rows[:, 1] * 4.0 > 0).astype(int)
        positives = np.flatnonzero(labels == 1).tolist()
        negatives = np.flatnonzero(labels == 0).tolist()
        ordered = rows[positives + negatives]
        ordered_labels = labels[positives + negatives]
        scaler = StandardScaler().fit(ordered)
        probe = LogisticRegression(max_iter=1000, random_state=42).fit(
            scaler.transform(ordered), ordered_labels
        )
        raw_normal = probe.coef_[0] / scaler.scale_
        raw_intercept = probe.intercept_[0] - scaler.mean_ @ raw_normal
        expected = l2_normalize(raw_normal) if normalize else raw_normal
        vector = LinearProbeExtractor.extract(
            [[[row]] for row in rows], positives, negatives, normalize=normalize
        )
        np.testing.assert_allclose(vector.directions[0], expected, rtol=1e-6, atol=1e-7)
        if not normalize:
            np.testing.assert_allclose(
                rows @ vector.directions[0] + raw_intercept,
                probe.decision_function(scaler.transform(rows)),
                rtol=1e-5,
                atol=1e-6,
            )
        assert vector.metadata["coordinate_space"] == "raw_activations"


class TestITILayouts:
    def test_infers_heads_and_preserves_component(self):
        train, validation = TestITIExtraction.captures()
        for capture in (train, validation):
            capture.layouts = {10: {"width": 4, "num_heads": 2, "head_size": 2}}
            capture.component = "attention_heads"
            capture.model = "test-model"
        vector = ITIExtractor.extract(
            train,
            [0, 1],
            validation_hidden_states=validation,
            validation_positive_indices=[0, 1],
        )
        assert vector.metadata["num_heads"] == {10: 2}
        assert vector.component == "attention_heads"
        assert vector.model_type == "test-model"
        with pytest.raises(ValueError, match="disagrees"):
            ITIExtractor.extract(
                train,
                [0, 1],
                validation_hidden_states=validation,
                validation_positive_indices=[0, 1],
                num_heads={10: 4},
            )

    def test_requires_attention_component(self):
        train, validation = TestITIExtraction.captures()
        train.component = "hidden_states"
        with pytest.raises(ValueError, match="attention_heads"):
            ITIExtractor.extract(
                train,
                [0, 1],
                validation_hidden_states=validation,
                validation_positive_indices=[0, 1],
                num_heads={10: 2},
            )

    def test_visits_one_layer_per_split_then_only_selected_layers(self):
        train, validation = TestITIExtraction.captures()
        visits = []

        class TrackedCapture(FakeCapture):
            def __init__(self, source, split):
                super().__init__(
                    [[[np.ones(4)], source.sample_rows(i, 10)] for i in range(4)],
                    [10, 20],
                )
                self.split = split

            def token(self, sample, layer, position=-1):
                visits.append((self.split, layer))
                return super().token(sample, layer, position)

        train, validation = (
            TrackedCapture(train, "train"),
            TrackedCapture(validation, "val"),
        )
        vector = ITIExtractor.extract(
            train,
            [0, 1],
            validation_hidden_states=validation,
            validation_positive_indices=[0, 1],
            num_heads={10: 2, 20: 2},
        )
        assert list(vector.directions) == [20]
        assert visits == [
            visit
            for group in [
                ("train", 10),
                ("val", 10),
                ("train", 20),
                ("val", 20),
                ("train", 20),
                ("val", 20),
            ]
            for visit in [group] * 4
        ]


def test_tensor_like_conversion_without_dtype_keeps_the_float_contract():
    from easysteer.extraction.selection import _to_numpy

    class TensorLike:
        def __init__(self):
            self.converted = False

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            self.converted = True
            return self

        def numpy(self):
            assert self.converted
            return np.array([1, 2], dtype=np.float32)

    value = TensorLike()
    np.testing.assert_array_equal(_to_numpy(value), [1, 2])
    assert value.converted
