"""
Difference of Means Extractor
"""

import numpy as np
from tqdm.auto import tqdm
from .utils import (
    StatisticalControlVector,
    derive_negative_indices,
    extract_token_hiddens,
)


class DiffMeanExtractor:
    """Difference of means method for control vector extraction"""
    
    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        model_type: str = "unknown",
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
            model_type (str): Model type name recorded in the result.
            normalize (bool): Normalize each direction to unit L2 norm.
            token_pos (int | str): Token position, -1 selects the last
                token (default); supports int/"first"/"last"/"mean"/
                "max"/"min".
            **kwargs (Any): Ignored here; unified_interface rejects
                unknown options before dispatching.

        Returns:
            StatisticalControlVector: The extracted control vector.
        """
        if negative_indices is None:
            negative_indices = derive_negative_indices(
                len(all_hidden_states), positive_indices
            )

        positive_hiddens, negative_hiddens = extract_token_hiddens(
            all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
        )

        directions = {}

        for layer in tqdm(positive_hiddens.keys(), desc="Computing DiffMean directions"):
            mean_positive = np.mean(positive_hiddens[layer], axis=0)
            mean_negative = np.mean(negative_hiddens[layer], axis=0)
            direction = mean_positive - mean_negative

            if normalize:
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm

            directions[layer] = direction.astype(np.float32)

        metadata = {
            "normalize": normalize,
            "token_pos": token_pos,
            "n_positive": len(positive_indices),
            "n_negative": len(negative_indices)
        }
        
        return StatisticalControlVector(
            model_type=model_type,
            method="diffmean",
            directions=directions,
            metadata=metadata
        ) 

    @staticmethod
    def from_moments(
        pos_moments,
        neg_moments,
        model_type: str = "unknown",
        normalize: bool = True,
    ) -> StatisticalControlVector:
        """Build a diffmean vector from streaming moment accumulators.

        pos_moments/neg_moments are MomentsAccumulator instances
        (easysteer.steer.accumulators) fed per-category rows layer by
        layer; equivalent to extract() on the same rows.
        """
        directions = {}
        layers = sorted(set(pos_moments.layers) & set(neg_moments.layers))
        if not layers:
            raise ValueError("accumulators share no layers")
        for layer in layers:
            d = pos_moments.mean(layer) - neg_moments.mean(layer)
            if normalize:
                norm = np.linalg.norm(d)
                if norm > 0:
                    d = d / norm
            directions[layer] = d.astype(np.float32)
        return StatisticalControlVector(
            model_type=model_type,
            method="diffmean",
            directions=directions,
            metadata={
                "normalized": normalize,
                "num_positive": int(max(pos_moments.count.values())),
                "num_negative": int(max(neg_moments.count.values())),
                "streaming": True,
            },
        )
