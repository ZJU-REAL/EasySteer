# SPDX-License-Identifier: Apache-2.0
"""Streaming accumulators for vector construction.

Cross-sample aggregation lives client-side by design (the engine
capture layer never computes corpus statistics — see
CAPTURE_REDESIGN_PROPOSAL.md §3.4). These accumulators consume
per-sample rows incrementally, so client memory stays O(1) while the
corpus streams through in chunks; `from_moments` constructors on the
extractors turn the accumulated statistics into control vectors.
"""

from typing import Dict, Optional

import numpy as np


class MomentsAccumulator:
    """Running (count, Σx, Σxxᵀ) per layer.

    Sufficient statistics for means, covariances, and standard PCA.
    ``track_second_moment=False`` keeps only count/Σx (enough for
    diffmean and category means).
    """

    def __init__(self, track_second_moment: bool = False):
        self.track_second_moment = track_second_moment
        self.count: Dict[int, int] = {}
        self.sum: Dict[int, np.ndarray] = {}
        self.sum_outer: Dict[int, np.ndarray] = {}

    def update(self, layer: int, rows) -> None:
        """Add rows (n, dim) of one layer (torch tensor or ndarray)."""
        x = np.asarray(
            rows.detach().to(dtype=__import__("torch").float32).cpu().numpy()
            if hasattr(rows, "detach")
            else rows,
            dtype=np.float64,
        )
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[0] == 0:
            return
        self.count[layer] = self.count.get(layer, 0) + x.shape[0]
        if layer in self.sum:
            self.sum[layer] += x.sum(axis=0)
        else:
            self.sum[layer] = x.sum(axis=0)
        if self.track_second_moment:
            outer = x.T @ x
            if layer in self.sum_outer:
                self.sum_outer[layer] += outer
            else:
                self.sum_outer[layer] = outer

    def mean(self, layer: int) -> np.ndarray:
        if self.count.get(layer, 0) == 0:
            raise ValueError(f"no rows accumulated for layer {layer}")
        return self.sum[layer] / self.count[layer]

    def covariance(self, layer: int) -> np.ndarray:
        if not self.track_second_moment:
            raise ValueError(
                "covariance requires track_second_moment=True"
            )
        n = self.count.get(layer, 0)
        if n < 2:
            raise ValueError(f"need >=2 rows for covariance, layer {layer}")
        mu = self.mean(layer)
        return self.sum_outer[layer] / n - np.outer(mu, mu)

    @property
    def layers(self):
        return sorted(self.count)


class DiffMeanAccumulator:
    """Streaming mean(pos) - mean(neg) per layer."""

    def __init__(self):
        self.pos = MomentsAccumulator()
        self.neg = MomentsAccumulator()

    def update(self, layer: int, rows, positive: bool) -> None:
        (self.pos if positive else self.neg).update(layer, rows)

    def direction(self, layer: int, normalize: bool = True) -> np.ndarray:
        d = self.pos.mean(layer) - self.neg.mean(layer)
        if normalize:
            norm = np.linalg.norm(d)
            if norm > 0:
                d = d / norm
        return d.astype(np.float32)


class TopKCountAccumulator:
    """Streaming per-expert top-k selection counts for router logits."""

    def __init__(self, top_k: int):
        self.top_k = top_k
        self.counts: Dict[int, np.ndarray] = {}
        self.tokens: Dict[int, int] = {}

    def update(self, layer: int, logits) -> None:
        """Add router logits (n_tokens, n_experts) of one layer."""
        x = np.asarray(
            logits.detach().float().cpu().numpy()
            if hasattr(logits, "detach")
            else logits,
            dtype=np.float32,
        )
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[0] == 0:
            return
        top = np.argpartition(-x, self.top_k - 1, axis=1)[:, : self.top_k]
        if layer not in self.counts:
            self.counts[layer] = np.zeros(x.shape[1], dtype=np.int64)
            self.tokens[layer] = 0
        np.add.at(self.counts[layer], top.reshape(-1), 1)
        self.tokens[layer] += x.shape[0]

    def rates(self, layer: int) -> np.ndarray:
        if self.tokens.get(layer, 0) == 0:
            raise ValueError(f"no tokens accumulated for layer {layer}")
        return self.counts[layer] / self.tokens[layer]
