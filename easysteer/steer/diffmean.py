"""
Difference of Means Extractor
"""

import numpy as np

from .accumulators import DiffMeanAccumulator
from .base_extractor import BaseExtractor
from .utils import (
    StatisticalControlVector,
    _metadata,
    derive_negative_indices,
)


class DiffMeanExtractor(BaseExtractor):
    """Difference of means method for control vector extraction"""

    method = "diffmean"
    progress_desc = "Computing DiffMean directions"

    @staticmethod
    def _direction(pos_rows, neg_rows, *, layer):
        """Mean positive activation minus mean negative activation.

        Args:
            pos_rows (np.ndarray): Positive activations, `(n_pos, dim)`.
            neg_rows (np.ndarray): Negative activations, `(n_neg, dim)`.
            layer (int): Layer key (unused; uniform hook signature).

        Returns:
            tuple[np.ndarray, dict]: The mean-difference direction and
                no per-layer extras.
        """
        del layer  # DiffMean has no per-layer state to report.
        return np.mean(pos_rows, axis=0) - np.mean(neg_rows, axis=0), {}

    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        normalize: bool = True,
        token_pos: int | str = -1,
        **kwargs
    ) -> StatisticalControlVector:
        """Extract control vectors using the difference-of-means method.

        Args:
            all_hidden_states (list | CaptureResult): Nested
                `[sample][layer][token]` hidden states, or a
                CaptureResult from easysteer.hidden_states.
            positive_indices (list[int]): Indices of positive samples.
                Never modified or rebound.
            negative_indices (list[int] | None): Indices of negative
                samples. If None, every sample index not in
                ``positive_indices`` becomes a negative, in ascending
                sample order (the convention shared by all extractors).

        Returns:
            StatisticalControlVector: The extracted control vector.
        """
        if negative_indices is None:
            negative_indices = derive_negative_indices(
                len(all_hidden_states), positive_indices
            )
        return DiffMeanExtractor._extract_template(
            all_hidden_states,
            positive_indices,
            negative_indices,
            normalize=normalize,
            token_pos=token_pos,
        )

    @staticmethod
    def from_moments(
        pos_moments,
        neg_moments,
        normalize: bool = True,
    ) -> StatisticalControlVector:
        """Build a diffmean vector from streaming moment accumulators.

        Equivalent to `extract()` on the same rows; the per-layer math
        delegates to `DiffMeanAccumulator.direction`.

        Args:
            pos_moments (MomentsAccumulator): Accumulator fed the
                positive rows layer by layer
                (easysteer.steer.accumulators).
            neg_moments (MomentsAccumulator): Accumulator fed the
                negative rows layer by layer.

        Returns:
            StatisticalControlVector: The extracted control vector.

        Raises:
            ValueError: If the accumulators share no layers.
        """
        layers = sorted(set(pos_moments.layers) & set(neg_moments.layers))
        if not layers:
            raise ValueError("accumulators share no layers")
        accumulator = DiffMeanAccumulator()
        accumulator.pos = pos_moments
        accumulator.neg = neg_moments
        directions = {
            layer: accumulator.direction(layer, normalize=normalize)
            for layer in layers
        }
        return StatisticalControlVector(
            method="diffmean",
            directions=directions,
            metadata=_metadata(
                normalize=normalize,
                n_positive=int(max(pos_moments.count.values())),
                n_negative=int(max(neg_moments.count.values())),
                streaming=True,
            ),
        )
