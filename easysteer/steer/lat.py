"""
Linear Algebraic Technique (LAT) Extractor
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from .utils import StatisticalControlVector

import logging
logger = logging.getLogger(__name__)


class LATExtractor:
    """Linear Algebraic Technique (LAT) method for control vector extraction"""
    
    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        model_type: str = "unknown",
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
            model_type (str): Model type name recorded in the result.
            n_components (int): Number of PCA components to fit; only
                the first component is used as the direction.
            use_positive_only (bool): Use only the positive samples.
            correct_direction (bool): Flip the vector if needed so it
                points from the negative samples toward the positive
                samples.
            normalize (bool): Normalize each direction to unit L2 norm.
            token_pos (int | str): Token position, -1 selects the last
                token (default); supports int/"first"/"last"/"mean"/
                "max"/"min".
            **kwargs (Any): Ignored here; unified_interface rejects
                unknown options before dispatching.

        Returns:
            StatisticalControlVector: The extracted control vector.
        """
        from .utils import derive_negative_indices, extract_token_hiddens

        if use_positive_only:
            sample_indices = positive_indices
        else:
            if negative_indices is None:
                negative_indices = derive_negative_indices(
                    len(all_hidden_states), positive_indices
                )
            sample_indices = positive_indices + negative_indices

        # LAT needs enough samples to form pairs for a meaningful PCA
        total_samples = len(sample_indices)
        min_pairs_needed = 2  # PCA needs at least 2 pairs
        max_pairs_possible = total_samples // 2

        if total_samples < 4:
            raise ValueError(
                f"The LAT method needs at least 4 samples to produce "
                f"2 difference pairs for a valid PCA, but only "
                f"{total_samples} were provided. Add more samples or "
                f"use another method (e.g. DiffMean)."
            )

        if max_pairs_possible < min_pairs_needed:
            raise ValueError(
                f"The LAT method needs at least {min_pairs_needed} "
                f"sample pairs for PCA, but {total_samples} samples "
                f"can produce at most {max_pairs_possible}. Add more "
                f"samples or use another method."
            )

        logger.info(
            f"LAT: using {total_samples} samples, producing "
            f"{max_pairs_possible} difference pairs"
        )

        directions = {}
        explained_variance = {}

        if use_positive_only:
            positive_hiddens, _ = extract_token_hiddens(
                all_hidden_states, sample_indices, [], token_pos=token_pos
            )
        else:
            positive_hiddens, negative_hiddens = extract_token_hiddens(
                all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
            )
            # Merge positive and negative samples
            combined_hiddens = {}
            for layer in positive_hiddens.keys():
                combined_hiddens[layer] = np.vstack([positive_hiddens[layer], negative_hiddens[layer]])
            positive_hiddens = combined_hiddens
        
        for layer in tqdm(list(positive_hiddens.keys()), desc="Computing LAT directions"):
            all_activations = positive_hiddens[layer]
            
            # LAT: pair activations at random and take differences
            logger.info(f"Layer {layer}: Shuffling {all_activations.shape[0]} activations")
            # Permute a copy: never mutate the caller-visible array.
            all_activations = np.random.permutation(all_activations)
            length = all_activations.shape[0] // 2
            differences = all_activations[:length] - all_activations[length:length * 2]
            
            logger.info(f"Layer {layer}: Shuffled and diff'd: {differences.shape[0]} pairs")
            logger.info(f"Layer {layer}: Potential NaNs: {np.isnan(differences).sum()}")
            logger.info(f"Layer {layer}: Potential Infs: {np.isinf(differences).sum()}")
            logger.info(f"Layer {layer}: Range: {differences.min()} to {differences.max()}")
            
            # Normalize the differences, guarding against zero norms
            norms = np.linalg.norm(differences, axis=1, keepdims=True)
            differences = np.where(norms == 0, 0, differences / norms)

            pca = PCA(n_components=min(n_components + 1, differences.shape[0], differences.shape[1]))
            pca.fit(differences)

            first_component = pca.components_[0]
            variance_explained = pca.explained_variance_ratio_[0]
            
            logger.info(f"Layer {layer}: LAT explains {variance_explained:.5%} of the variance")
            
            # Direction correction (point from negatives toward positives)
            if correct_direction and not use_positive_only and negative_indices is not None and len(negative_indices) > 0:
                # Re-extract positive/negative activations for correction
                pos_hiddens_orig, neg_hiddens_orig = extract_token_hiddens(
                    all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
                )

                pos_activations_layer = pos_hiddens_orig[layer]
                neg_activations_layer = neg_hiddens_orig[layer]

                vec_norm = np.linalg.norm(first_component)
                if vec_norm > 1e-6:  # Avoid division by zero
                    # Project activations onto the principal component
                    proj_pos = (pos_activations_layer @ first_component) / vec_norm
                    proj_neg = (neg_activations_layer @ first_component) / vec_norm

                    # A lower mean positive projection means the
                    # direction is inverted; flip it.
                    if np.mean(proj_pos) < np.mean(proj_neg):
                        first_component *= -1
                        logger.info(f"Layer {layer}: Direction corrected (flipped)")
            
            if normalize:
                norm = np.linalg.norm(first_component)
                if norm > 0:
                    first_component = first_component / norm
            
            directions[layer] = first_component.astype(np.float32)
            explained_variance[layer] = float(variance_explained)
        
        metadata = {
            "normalize": normalize,
            "n_components": n_components,
            "use_positive_only": use_positive_only,
            "correct_direction": correct_direction,
            "token_pos": token_pos,
            "n_samples": len(sample_indices),
            "n_positive": len(positive_indices),
            "n_negative": len(negative_indices) if negative_indices else 0,
            "explained_variance": explained_variance
        }
        
        return StatisticalControlVector(
            model_type=model_type,
            method="lat",
            directions=directions,
            metadata=metadata
        ) 