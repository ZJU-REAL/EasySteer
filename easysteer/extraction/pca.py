# SPDX-License-Identifier: Apache-2.0
"""Principal-component control-vector extraction."""

import logging

import numpy as np
from sklearn.decomposition import PCA

from ._utils import _metadata, correct_sign, l2_normalize
from .base import BaseExtractor
from .result import StatisticalControlVector
from .selection import validate_sample_groups

logger = logging.getLogger(__name__)

_PCA_LOG_LABELS = {
    "standard": "PCA",
    "center": "PCA on centered data",
    "diff": "PCA on differences",
}


def _fit_first_component(activations):
    """Fit `PCA(n_components=1)` and return its first component.

    Args:
        activations (np.ndarray): Row matrix to decompose,
            `(n_rows, dim)`.

    Returns:
        tuple[np.ndarray, float]: The first principal component and
            its explained-variance ratio.
    """
    pca = PCA(n_components=1)
    pca.fit(activations)
    return pca.components_[0], float(pca.explained_variance_ratio_[0])


class PCAExtractor(BaseExtractor):
    """Extract directions with principal component analysis."""

    method = "pca"
    progress_desc = "Computing PCA directions"

    @staticmethod
    def _direction(pos_rows, neg_rows, *, layer, method, correct_direction):
        """First principal component of the variant-specific rows.

        Args:
            pos_rows (np.ndarray): Positive activations, `(n_pos, dim)`.
            neg_rows (np.ndarray | None): Negative activations,
                `(n_neg, dim)`; optional for the "standard" variant.
            layer (int): Layer key, used for logging.
            method (str): PCA variant: "standard", "diff" or "center".
            correct_direction (bool): Flip the component if needed so
                it points from the negative toward the positive
                samples when negatives are available.

        Returns:
            tuple[np.ndarray, dict]: The component and its explained
                variance under the `"explained_variance"` key.
        """
        if method == "standard":
            activations = (
                pos_rows if isinstance(pos_rows, np.ndarray) else np.vstack(pos_rows)
            )
        elif method == "center":
            # Centered PCA: truncate to equal numbers of positives and
            # negatives, then center each pair on its midpoint.
            min_samples = min(len(pos_rows), len(neg_rows))
            pos = pos_rows[:min_samples]
            neg = neg_rows[:min_samples]
            centers = (pos + neg) / 2
            activations = np.vstack([pos - centers, neg - centers])
        else:  # "diff" — extract() validated the variant name.
            # Difference of each positive/negative pair.
            min_samples = min(len(pos_rows), len(neg_rows))
            differences = [pos_rows[i] - neg_rows[i] for i in range(min_samples)]
            # Extra positives are paired with the negative mean.
            if len(pos_rows) > min_samples:
                neg_mean = np.mean(neg_rows, axis=0)
                differences.extend(row - neg_mean for row in pos_rows[min_samples:])
            # Extra negatives are paired with the positive mean.
            if len(neg_rows) > min_samples:
                pos_mean = np.mean(pos_rows, axis=0)
                differences.extend(pos_mean - row for row in neg_rows[min_samples:])
            activations = np.vstack(differences)

        component, variance = _fit_first_component(activations)
        logger.info(
            f"Layer {layer}: {_PCA_LOG_LABELS[method]} explains "
            f"{variance:.5%} of the variance"
        )

        if correct_direction and neg_rows is not None:
            component = correct_sign(component, pos_rows, neg_rows)

        return component, {"explained_variance": variance}

    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        n_components: int = 1,
        method: str | None = None,
        correct_direction: bool = True,
        normalize: bool = True,
        token_pos: int | str = -1,
        *,
        variant: str | None = None,
    ) -> StatisticalControlVector:
        """Extract control vectors using the PCA method.

        Args:
            all_hidden_states (list | CaptureResult): Nested
                `[sample][layer][token]` hidden states, or a
                CaptureResult from easysteer.capture.
            positive_indices (list[int]): Indices of positive samples.
                Never modified or rebound.
            negative_indices (list[int] | None): Indices of negative
                samples. If None, every sample index not in
                ``positive_indices`` becomes a negative, in ascending
                sample order (the convention shared by all extractors).

        Returns:
            StatisticalControlVector: The extracted control vector.

        ``variant`` selects "standard", "diff", or "center". The old
        ``method`` argument remains an alias. Standard PCA fits positives
        only; negatives, when supplied or derivable, determine its sign.
        """
        if method is not None and variant is not None and method != variant:
            raise ValueError("PCA method and variant must agree when both are supplied")
        method = variant if variant is not None else method
        if method is None:
            method = "standard"
        supported_methods = ("standard", "diff", "center")
        if method not in supported_methods:
            raise ValueError(
                f"Unknown PCA method: {method!r}. Supported methods: "
                f"{list(supported_methods)}"
            )
        if n_components != 1:
            raise ValueError(
                f"n_components={n_components} is not supported; "
                f"PCAExtractor only extracts the first principal "
                f"component (n_components=1)"
            )

        positive_indices, negative_indices = validate_sample_groups(
            len(all_hidden_states),
            positive_indices,
            negative_indices,
            require_negative=method != "standard",
        )
        if method == "standard" and len(positive_indices) < 2:
            raise ValueError("standard PCA requires at least two positive samples")
        if method == "diff" and max(len(positive_indices), len(negative_indices)) < 2:
            raise ValueError("difference PCA requires at least two sample pairs")
        extraction_negatives = (
            negative_indices if method != "standard" or correct_direction else []
        )

        return PCAExtractor._extract_template(
            all_hidden_states,
            positive_indices,
            negative_indices,
            normalize=normalize,
            token_pos=token_pos,
            extraction_negatives=extraction_negatives,
            opts={"method": method, "correct_direction": correct_direction},
            extra_metadata={
                "n_components": 1,
                "method": method,
                "variant": method,
                "correct_direction": correct_direction,
            },
            method=f"pca_{method}",
        )

    @staticmethod
    def from_moments(
        moments,
        normalize: bool = True,
        pos_moments=None,
        neg_moments=None,
    ) -> StatisticalControlVector:
        """Standard PCA from a streaming second-moment accumulator.

        The direction per layer is the top eigenvector of the
        covariance. When ``pos_moments``/``neg_moments`` are given, the
        sign is corrected so the direction points from the negative
        mean to the positive mean, matching
        `extract(method="standard", correct_direction=True)`.

        Args:
            moments (MomentsAccumulator): Accumulator with
                ``track_second_moment=True`` fed the positive rows.

        Returns:
            StatisticalControlVector: The extracted control vector.

        Raises:
            ValueError: If ``moments`` holds no layers.
        """
        directions = {}
        variance = {}
        if not moments.layers:
            raise ValueError("accumulator holds no layers")
        for layer in moments.layers:
            cov = moments.covariance(layer)
            eigvals, eigvecs = np.linalg.eigh(cov)
            direction = eigvecs[:, -1]
            total = float(eigvals.sum())
            variance[layer] = float(eigvals[-1] / total) if total > 0 else 0.0
            if pos_moments is not None and neg_moments is not None:
                gap = pos_moments.mean(layer) - neg_moments.mean(layer)
                if float(gap @ direction) < 0:
                    direction = -direction
            if normalize:
                direction = l2_normalize(direction)
            directions[layer] = direction.astype(np.float32)
        return StatisticalControlVector(
            method="pca_standard",
            directions=directions,
            metadata=_metadata(
                normalize=normalize,
                n_positive=int(max(moments.count.values())),
                n_negative=(
                    int(max(neg_moments.count.values()))
                    if neg_moments is not None
                    else 0
                ),
                explained_variance=variance,
                streaming=True,
            ),
        )
