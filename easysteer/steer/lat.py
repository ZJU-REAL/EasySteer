"""
Linear Algebraic Technique (LAT) Extractor
"""

import logging

import numpy as np
from sklearn.decomposition import PCA

from .base_extractor import BaseExtractor
from .utils import (
    StatisticalControlVector,
    correct_sign,
    derive_negative_indices,
)

logger = logging.getLogger(__name__)


class LATExtractor(BaseExtractor):
    """Linear Algebraic Technique (LAT) method for control vector extraction"""

    method = "lat"
    progress_desc = "Computing LAT directions"

    @staticmethod
    def _direction(pos_rows, neg_rows, *, layer, n_components,
                   correct_direction):
        """PCA over normalized differences of randomly paired rows.

        Args:
            pos_rows (np.ndarray): Positive activations, `(n_pos, dim)`.
            neg_rows (np.ndarray | None): Negative activations,
                `(n_neg, dim)`, or None in positive-only mode; when
                present they join the pairing pool and drive the
                direction correction.
            layer (int): Layer key, used for logging.
            n_components (int): Number of PCA components to fit; only
                the first component is used as the direction.
            correct_direction (bool): Flip the component if needed so
                it points from the negative toward the positive
                samples (requires negatives).

        Returns:
            tuple[np.ndarray, dict]: The component and its explained
                variance under the `"explained_variance"` key.
        """
        activations = (
            pos_rows if neg_rows is None else np.vstack([pos_rows, neg_rows])
        )

        # LAT: pair activations at random and take differences
        logger.info(
            f"Layer {layer}: Shuffling {activations.shape[0]} activations"
        )
        # Permute a copy: never mutate the caller-visible array.
        activations = np.random.permutation(activations)
        length = activations.shape[0] // 2
        differences = activations[:length] - activations[length : length * 2]

        logger.info(
            f"Layer {layer}: Shuffled and diff'd: {differences.shape[0]} pairs"
        )
        logger.info(
            f"Layer {layer}: Potential NaNs: {np.isnan(differences).sum()}"
        )
        logger.info(
            f"Layer {layer}: Potential Infs: {np.isinf(differences).sum()}"
        )
        logger.info(
            f"Layer {layer}: Range: {differences.min()} to "
            f"{differences.max()}"
        )

        # Normalize the differences, guarding against zero norms
        norms = np.linalg.norm(differences, axis=1, keepdims=True)
        differences = np.where(norms == 0, 0, differences / norms)

        pca = PCA(
            n_components=min(
                n_components + 1, differences.shape[0], differences.shape[1]
            )
        )
        pca.fit(differences)

        component = pca.components_[0]
        variance = float(pca.explained_variance_ratio_[0])
        logger.info(
            f"Layer {layer}: LAT explains {variance:.5%} of the variance"
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
        use_positive_only: bool = True,
        correct_direction: bool = True,
        normalize: bool = True,
        token_pos: int | str = -1,
        **kwargs
    ) -> StatisticalControlVector:
        """Extract control vectors using the LAT method.

        LAT is PCA over normalized differences of random pairs of
        activations.

        Args:
            all_hidden_states (list | CaptureResult): Nested
                `[sample][layer][token]` hidden states, or a
                CaptureResult from easysteer.hidden_states.
            positive_indices (list[int]): Indices of positive samples.
                Never modified or rebound.
            negative_indices (list[int] | None): Indices of negative
                samples. If None and ``use_positive_only`` is False,
                every sample index not in ``positive_indices`` becomes
                a negative, in ascending sample order (the convention
                shared by all extractors).

        Returns:
            StatisticalControlVector: The extracted control vector.
        """
        if use_positive_only:
            n_samples = len(positive_indices)
            extraction_negatives = []  # Negatives never enter the pool.
        else:
            if negative_indices is None:
                negative_indices = derive_negative_indices(
                    len(all_hidden_states), positive_indices
                )
            n_samples = len(positive_indices) + len(negative_indices)
            extraction_negatives = None  # Extract the negatives too.

        # LAT needs enough samples to form pairs for a meaningful PCA:
        # 4 samples yield the minimum of 2 difference pairs.
        if n_samples < 4:
            raise ValueError(
                f"The LAT method needs at least 4 samples to produce "
                f"2 difference pairs for a valid PCA, but only "
                f"{n_samples} were provided. Add more samples or "
                f"use another method (e.g. DiffMean)."
            )

        logger.info(
            f"LAT: using {n_samples} samples, producing "
            f"{n_samples // 2} difference pairs"
        )

        return LATExtractor._extract_template(
            all_hidden_states,
            positive_indices,
            negative_indices,
            normalize=normalize,
            token_pos=token_pos,
            extraction_negatives=extraction_negatives,
            opts={
                "n_components": n_components,
                "correct_direction": correct_direction,
            },
            extra_metadata={
                "n_components": n_components,
                "use_positive_only": use_positive_only,
                "correct_direction": correct_direction,
                "n_samples": n_samples,
            },
        )
