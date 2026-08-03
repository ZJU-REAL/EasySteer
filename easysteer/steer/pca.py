"""
Principal Component Analysis Extractor
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from .utils import StatisticalControlVector

import logging
logger = logging.getLogger(__name__)


class PCAExtractor:
    """Principal Component Analysis method for control vector extraction"""
    
    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        model_type: str = "unknown",
        n_components: int = 1,
        method: str = "standard",  # "standard", "diff", or "center"
        correct_direction: bool = True,
        normalize: bool = True,
        token_pos: int | str = -1,
        **kwargs
    ) -> StatisticalControlVector:
        """Extract control vectors using the PCA method.

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
            n_components (int): Number of PCA components. Only 1 is
                supported; other values raise ValueError.
            method (str): PCA variant, one of:
                "standard" - plain PCA (positive samples only),
                "diff" - PCA over positive/negative differences,
                "center" - PCA over pair-centered samples.
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
        supported_methods = ("standard", "diff", "center")
        if method not in supported_methods:
            raise ValueError(
                f"Unknown PCA method: {method!r}. Supported methods: "
                f"{list(supported_methods)}"
            )
        # Only the first principal component is ever computed; reject
        # anything else instead of silently recording an unused value.
        if n_components != 1:
            raise ValueError(
                f"n_components={n_components} is not supported; "
                f"PCAExtractor only extracts the first principal "
                f"component (n_components=1)"
            )

        from .utils import derive_negative_indices, extract_token_hiddens

        if method in ["diff", "center"]:
            if negative_indices is None:
                negative_indices = derive_negative_indices(
                    len(all_hidden_states), positive_indices
                )
            if len(negative_indices) == 0:
                raise ValueError(
                    f"PCA method {method!r} requires negative samples, "
                    f"but none were provided or derivable"
                )

        directions = {}
        explained_variance = {}

        positive_hiddens, negative_hiddens = extract_token_hiddens(
            all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
        )

        for layer in tqdm(list(positive_hiddens.keys()), desc="Computing PCA directions"):
            if method == "standard":
                # Plain PCA over positive samples only
                all_activations = positive_hiddens[layer]

                if not isinstance(all_activations, np.ndarray):
                    all_activations = np.vstack(all_activations)

                pca = PCA(n_components=1)
                pca.fit(all_activations)

                first_component = pca.components_[0]
                variance_explained = pca.explained_variance_ratio_[0]

                logger.info(f"Layer {layer}: PCA explains {variance_explained:.5%} of the variance")

            elif method == "center":
                # Centered PCA (requires positive and negative samples)
                pos_activations = positive_hiddens[layer]  # [n_pos, hidden_dim]
                neg_activations = negative_hiddens[layer]  # [n_neg, hidden_dim]

                # Truncate to equal numbers of positives and negatives
                min_samples = min(len(pos_activations), len(neg_activations))
                pos_activations = pos_activations[:min_samples]
                neg_activations = neg_activations[:min_samples]

                # Center each pair on its positive/negative midpoint
                centers = (pos_activations + neg_activations) / 2

                centered_pos = pos_activations - centers
                centered_neg = neg_activations - centers

                all_activations = np.vstack([centered_pos, centered_neg])

                pca = PCA(n_components=1)
                pca.fit(all_activations)

                first_component = pca.components_[0]
                variance_explained = pca.explained_variance_ratio_[0]

                logger.info(f"Layer {layer}: PCA on centered data explains {variance_explained:.5%} of the variance")

            elif method == "diff":
                # Difference PCA (requires positive and negative samples)
                pos_activations = positive_hiddens[layer]  # [n_pos, hidden_dim]
                neg_activations = negative_hiddens[layer]  # [n_neg, hidden_dim]

                # Difference of each positive/negative pair
                min_samples = min(len(pos_activations), len(neg_activations))
                differences = []

                for i in range(min_samples):
                    diff = pos_activations[i] - neg_activations[i]
                    differences.append(diff)

                # Extra positives are paired with the negative mean
                if len(pos_activations) > min_samples:
                    neg_mean = np.mean(neg_activations, axis=0)
                    for i in range(min_samples, len(pos_activations)):
                        diff = pos_activations[i] - neg_mean
                        differences.append(diff)

                # Extra negatives are paired with the positive mean
                if len(neg_activations) > min_samples:
                    pos_mean = np.mean(pos_activations, axis=0)
                    for i in range(min_samples, len(neg_activations)):
                        diff = pos_mean - neg_activations[i]
                        differences.append(diff)

                all_activations = np.vstack(differences)

                pca = PCA(n_components=1)
                pca.fit(all_activations)

                first_component = pca.components_[0]
                variance_explained = pca.explained_variance_ratio_[0]

                logger.info(f"Layer {layer}: PCA on differences explains {variance_explained:.5%} of the variance")

            # Direction correction (point from negatives toward positives)
            if correct_direction and method in ["diff", "center"]:
                pos_activations_layer = positive_hiddens[layer]
                neg_activations_layer = negative_hiddens[layer]

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
            "n_components": 1,
            "method": method,
            "correct_direction": correct_direction,
            "token_pos": token_pos,
            "n_positive": len(positive_indices),
            "n_negative": len(negative_indices) if negative_indices else 0,
            "explained_variance": explained_variance
        }
        
        return StatisticalControlVector(
            model_type=model_type,
            method=f"pca_{method}",
            directions=directions,
            metadata=metadata
        ) 

    @staticmethod
    def from_moments(
        moments,
        model_type: str = "unknown",
        normalize: bool = True,
        pos_moments=None,
        neg_moments=None,
    ) -> StatisticalControlVector:
        """Standard PCA from a streaming second-moment accumulator.

        moments must be a MomentsAccumulator(track_second_moment=True)
        fed the positive rows; the direction per layer is the top
        eigenvector of the covariance. When pos_moments/neg_moments
        (first-moment accumulators) are given, the sign is corrected so
        the direction points from the negative mean to the positive
        mean, matching extract(method="standard", correct_direction).
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
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
            directions[layer] = direction.astype(np.float32)
        return StatisticalControlVector(
            model_type=model_type,
            method="pca_standard",
            directions=directions,
            metadata={
                "normalized": normalize,
                "explained_variance": variance,
                "streaming": True,
            },
        )
