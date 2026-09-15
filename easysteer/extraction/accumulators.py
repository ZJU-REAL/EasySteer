# SPDX-License-Identifier: Apache-2.0
"""Streaming accumulators for vector construction.

Retain per-layer statistics instead of corpus rows. The extractors'
``from_moments`` methods turn these statistics into control vectors.
"""

from numbers import Integral

import numpy as np

from ._utils import l2_normalize

DEFAULT_WORKING_BYTES = 256 * 1024**2


def validate_memory_budget(value: int) -> int:
    """Require an explicit finite allocation budget."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("max_working_bytes must be a positive integer")
    return int(value)


def check_memory_budget(required: int, budget: int, *, operation: str) -> None:
    if required > budget:
        raise MemoryError(
            f"{operation} requires approximately {required:,} working bytes, "
            f"exceeding max_working_bytes={budget:,}. Reduce layers or batch "
            "size, use incremental_pca instead of exact covariance, or raise "
            "the budget explicitly."
        )


def _rows_shape(rows):
    """Inspect a Python row sequence without allocating its numeric matrix."""
    shape = getattr(rows, "shape", None)
    if shape is not None:
        return tuple(shape)
    if isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a numeric row sequence")
    try:
        count = len(rows)
        if count == 0:
            return (0,)
        first = rows[0]
    except (TypeError, KeyError) as exc:
        raise ValueError("rows must have shape (n, width) or (width,)") from exc
    try:
        width = len(first)
    except TypeError:
        return (count,)
    return (count, width)


def _validate_row_sequence(rows, shape):
    """Reject ragged/deeper nested input before NumPy can allocate it."""
    if getattr(rows, "shape", None) is not None:
        return
    for index in range(shape[0]):
        row = rows[index] if len(shape) == 2 else (rows[index],)
        if len(shape) == 2:
            try:
                width = len(row)
            except TypeError as exc:
                raise ValueError("rows must be rectangular") from exc
            if width != shape[-1]:
                raise ValueError("rows must be rectangular")
        for value in row:
            try:
                len(value)
            except TypeError:
                continue
            raise ValueError("rows must contain scalar numeric values")


class MomentsAccumulator:
    """Running (count, Σx, Σxxᵀ) per layer.

    Sufficient statistics for means, covariances, and standard PCA.
    ``track_second_moment=False`` keeps only count/Σx (enough for
    diffmean and category means). ``max_working_bytes`` estimates numerical
    array storage and scratch, excluding caller-owned rows and library overhead.
    Exact covariance is quadratic in width and is checked before any update.
    """

    def __init__(
        self,
        track_second_moment: bool = False,
        *,
        max_working_bytes: int = DEFAULT_WORKING_BYTES,
    ):
        self.track_second_moment = track_second_moment
        self.max_working_bytes = validate_memory_budget(max_working_bytes)
        self.count: dict[int, int] = {}
        self.sum: dict[int, np.ndarray] = {}
        self.sum_outer: dict[int, np.ndarray] = {}

    def update(self, layer: int, rows) -> None:
        """Add rows (n, dim) of one layer (torch tensor or ndarray)."""
        shape = _rows_shape(rows)
        if len(shape) not in (1, 2) or shape[-1] == 0:
            raise ValueError("rows must have shape (n, width) or (width,)")
        width = shape[-1]
        if layer in self.sum and self.sum[layer].shape != (width,):
            raise ValueError(f"layer {layer}: row width changed")
        n_rows = shape[0] if len(shape) == 2 else 1
        if n_rows == 0:
            return
        retained = sum(value.nbytes for value in self.sum.values()) + sum(
            value.nbytes for value in self.sum_outer.values()
        )
        # Reserve the cast, sum and product before allocating them or updating
        # statistics. Exact covariance and eigendecomposition need additional
        # square matrices; reserve four of them up front as well.
        extra = 8 * (2 * n_rows * width + 2 * width)
        if self.track_second_moment:
            largest_width = max(
                width, max((value.size for value in self.sum.values()), default=0)
            )
            extra += 8 * 4 * largest_width**2
        check_memory_budget(
            retained + extra, self.max_working_bytes, operation="moment accumulation"
        )
        _validate_row_sequence(rows, shape)
        x = np.asarray(
            rows.detach().cpu().double().numpy() if hasattr(rows, "detach") else rows,
            dtype=np.float64,
        ).reshape(n_rows, width)
        if not np.isfinite(x).all():
            raise ValueError("rows must contain only finite values")
        summed = x.sum(axis=0)
        outer = x.T @ x if self.track_second_moment else None
        if layer in self.sum:
            self.sum[layer] += summed
        else:
            self.sum[layer] = summed
        if outer is not None:
            if layer in self.sum_outer:
                self.sum_outer[layer] += outer
            else:
                self.sum_outer[layer] = outer
        self.count[layer] = self.count.get(layer, 0) + n_rows

    def mean(self, layer: int) -> np.ndarray:
        if self.count.get(layer, 0) == 0:
            raise ValueError(f"no rows accumulated for layer {layer}")
        return self.sum[layer] / self.count[layer]

    def covariance(self, layer: int) -> np.ndarray:
        if not self.track_second_moment:
            raise ValueError("covariance requires track_second_moment=True")
        n = self.count.get(layer, 0)
        if n < 2:
            raise ValueError(f"need >=2 rows for covariance, layer {layer}")
        width = self.sum[layer].size
        retained = sum(value.nbytes for value in self.sum.values()) + sum(
            value.nbytes for value in self.sum_outer.values()
        )
        check_memory_budget(
            retained + 8 * (4 * width**2 + 2 * width),
            self.max_working_bytes,
            operation="covariance and eigendecomposition",
        )
        mu = self.mean(layer)
        return self.sum_outer[layer] / n - np.outer(mu, mu)

    @property
    def layers(self):
        return sorted(self.count)


class DiffMeanAccumulator:
    """Streaming mean(pos) - mean(neg) per layer."""

    def __init__(self, *, max_working_bytes: int = DEFAULT_WORKING_BYTES):
        self.max_working_bytes = validate_memory_budget(max_working_bytes)
        self.pos = MomentsAccumulator(max_working_bytes=max_working_bytes)
        self.neg = MomentsAccumulator(max_working_bytes=max_working_bytes)

    def update(self, layer: int, rows, positive: bool) -> None:
        if not isinstance(positive, (bool, Integral, np.bool_)) or positive not in (
            0,
            1,
        ):
            raise ValueError("positive must be Boolean or integer 0/1")
        shape = _rows_shape(rows)
        if len(shape) not in (1, 2) or shape[-1] == 0:
            raise ValueError("rows must have shape (n, width) or (width,)")
        width = shape[-1]
        count = shape[0] if len(shape) == 2 else 1
        retained = sum(
            value.nbytes for acc in (self.pos, self.neg) for value in acc.sum.values()
        )
        check_memory_budget(
            retained + 8 * (2 * count * width + 2 * width),
            self.max_working_bytes,
            operation="diffmean accumulation",
        )
        (self.pos if positive else self.neg).update(layer, rows)

    def direction(self, layer: int, normalize: bool = True) -> np.ndarray:
        """One layer's mean(pos) - mean(neg) direction.

        Args:
            layer (int): Layer key to compute the direction for.
            normalize (bool): Normalize the direction to unit L2 norm.

        Returns:
            np.ndarray: The float32 direction vector.
        """
        d = self.pos.mean(layer) - self.neg.mean(layer)
        if normalize:
            d = l2_normalize(d)
        return d.astype(np.float32)


class TopKCountAccumulator:
    """Streaming per-expert top-k selection counts for router logits."""

    def __init__(self, top_k: int):
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.top_k = int(top_k)
        self.counts: dict[int, np.ndarray] = {}
        self.tokens: dict[int, int] = {}

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
        if x.ndim != 2 or x.shape[1] < self.top_k:
            raise ValueError(
                "logits must have shape (tokens, experts) with experts >= top_k"
            )
        if layer in self.counts and self.counts[layer].size != x.shape[1]:
            raise ValueError(f"layer {layer}: expert count changed")
        if not np.isfinite(x).all():
            raise ValueError("logits must contain only finite values")
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
