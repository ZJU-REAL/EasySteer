# SPDX-License-Identifier: Apache-2.0
"""Streaming accumulators must equal their batch counterparts.

The extraction/post-processing boundary keeps corpus statistics
client-side; these units pin the equivalence between chunked streaming
accumulation and the one-shot extractors on identical data.
"""

import numpy as np
import pytest

from easysteer.steer import (
    DiffMeanExtractor,
    MomentsAccumulator,
    PCAExtractor,
    TopKCountAccumulator,
)

RNG = np.random.default_rng(7)
DIM = 32
N = 40


def chunks(x, k=7):
    for i in range(0, len(x), k):
        yield x[i : i + k]


class TestMomentsAccumulator:
    def test_mean_and_covariance_match_numpy(self):
        x = RNG.normal(size=(N, DIM))
        acc = MomentsAccumulator(track_second_moment=True)
        for part in chunks(x):
            acc.update(3, part)
        assert np.allclose(acc.mean(3), x.mean(axis=0))
        assert np.allclose(acc.covariance(3), np.cov(x.T, bias=True))

    def test_empty_layer_raises(self):
        acc = MomentsAccumulator()
        with pytest.raises(ValueError, match="no rows"):
            acc.mean(0)


class TestFromMoments:
    def test_diffmean_equals_batch_extractor(self):
        pos = RNG.normal(size=(N, DIM)) + 0.5
        neg = RNG.normal(size=(N, DIM))
        pos_acc, neg_acc = MomentsAccumulator(), MomentsAccumulator()
        for part in chunks(pos):
            pos_acc.update(5, part)
        for part in chunks(neg):
            neg_acc.update(5, part)
        streamed = DiffMeanExtractor.from_moments(pos_acc, neg_acc)

        nested = [[np.array([row])] for row in pos] + [
            [np.array([row])] for row in neg
        ]
        batch = DiffMeanExtractor.extract(
            nested,
            positive_indices=list(range(N)),
            negative_indices=list(range(N, 2 * N)),
        )
        # from_moments keys by true layer id (5); extract keys by
        # positional layer index (0) — same single layer's data.
        assert np.allclose(
            streamed.directions[5], batch.directions[0], atol=1e-5
        )

    def test_pca_standard_direction_matches_covariance_eig(self):
        # The two axes are orthogonal; the leading eigenvector is [3, 4] / 5.
        direction = np.array([0.6, 0.8])
        other = np.array([-0.8, 0.6])
        x = np.vstack([3 * direction, -3 * direction, other, -other])
        acc = MomentsAccumulator(track_second_moment=True)
        for part in chunks(x, k=3):
            acc.update(2, part)
        v = PCAExtractor.from_moments(acc).directions[2]
        np.testing.assert_allclose(v * np.sign(v @ direction), direction, atol=1e-6)

    def test_pca_sign_correction_from_category_means(self):
        direction = np.array([0.6, 0.8])
        x = np.array([-3, -1, 1, 3])[:, None] * direction
        acc = MomentsAccumulator(track_second_moment=True)
        acc.update(0, x)
        pos, neg = MomentsAccumulator(), MomentsAccumulator()
        pos.update(0, x + direction)
        neg.update(0, x - direction)
        forward = PCAExtractor.from_moments(acc, pos_moments=pos, neg_moments=neg)
        reverse = PCAExtractor.from_moments(acc, pos_moments=neg, neg_moments=pos)
        np.testing.assert_allclose(forward.directions[0], direction, atol=1e-6)
        np.testing.assert_allclose(reverse.directions[0], -direction, atol=1e-6)


class TestTopKCounts:
    def test_counts_match_direct_topk(self):
        logits = RNG.normal(size=(N, 16))
        acc = TopKCountAccumulator(top_k=4)
        for part in chunks(logits):
            acc.update(1, part)
        direct = np.zeros(16, dtype=np.int64)
        for row in logits:
            direct[np.argsort(-row)[:4]] += 1
        assert (acc.counts[1] == direct).all()
        assert np.isclose(acc.rates(1).sum(), 4.0)
