"""
Linear Probe Extractor
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from .utils import StatisticalControlVector

import logging
logger = logging.getLogger(__name__)


class LinearProbeExtractor:
    """Linear Probe method for control vector extraction"""
    
    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        model_type: str = "unknown",
        normalize: bool = True,
        token_pos: int | str = -1,
        regularization: str = "l2",
        C: float = 1.0,
        standardize: bool = True,
        **kwargs
    ) -> StatisticalControlVector:
        """Extract control vectors using the Linear Probe method.

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
            regularization (str): Regularization type, one of "l1",
                "l2", "elasticnet", "none"; anything else raises
                ValueError.
            C (float): Inverse regularization strength (smaller means
                stronger regularization).
            standardize (bool): Standardize features before fitting.
            **kwargs (Any): Ignored here; unified_interface rejects
                unknown options before dispatching.

        Returns:
            StatisticalControlVector: The extracted control vector.
        """
        from .utils import derive_negative_indices, extract_token_hiddens

        if negative_indices is None:
            negative_indices = derive_negative_indices(
                len(all_hidden_states), positive_indices
            )

        total_samples = len(positive_indices) + len(negative_indices)
        if total_samples < 4:
            raise ValueError(
                f"The LinearProbe method needs at least 4 samples "
                f"(2 positive + 2 negative), but only {total_samples} "
                f"were provided."
            )

        if len(positive_indices) < 1 or len(negative_indices) < 1:
            raise ValueError(
                f"The LinearProbe method needs at least 1 positive and "
                f"1 negative sample, but got {len(positive_indices)} "
                f"positive and {len(negative_indices)} negative."
            )

        penalty_map = {
            "l1": "l1",
            "l2": "l2",
            "elasticnet": "elasticnet",
            "none": None
        }
        if regularization not in penalty_map:
            raise ValueError(
                f"Unknown regularization: {regularization!r}. Accepted "
                f"values: {list(penalty_map)}"
            )
        penalty = penalty_map[regularization]

        if regularization == "l1" and C <= 1.0:
            logger.warning(
                f"L1 regularization with C={C} may be too strong and "
                f"can produce all-zero weights. Try a larger C (e.g. "
                f"C=10.0 or C=100.0) to reduce sparsification."
            )

        logger.info(
            f"LinearProbe: using {len(positive_indices)} positive and "
            f"{len(negative_indices)} negative samples"
        )
        logger.info(f"Regularization: {regularization}, C: {C}")

        directions = {}
        model_scores = {}

        positive_hiddens, negative_hiddens = extract_token_hiddens(
            all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
        )

        for layer in tqdm(
            list(positive_hiddens.keys()), desc="Computing LinearProbe directions"
        ):
            X_pos = positive_hiddens[layer]  # [n_positive, hidden_dim]
            X_neg = negative_hiddens[layer]  # [n_negative, hidden_dim]

            X = np.vstack([X_pos, X_neg])  # [n_total, hidden_dim]
            y = np.hstack([
                np.ones(len(X_pos)),   # positive samples labeled 1
                np.zeros(len(X_neg))   # negative samples labeled 0
            ])

            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            if penalty == "elasticnet":
                clf = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    l1_ratio=0.5,  # elasticnet l1/l2 mixing ratio
                    solver="saga",
                    max_iter=1000,
                    random_state=42
                )
            elif penalty is None:
                clf = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42
                )
            else:
                solver = "liblinear" if penalty == "l1" else "lbfgs"
                clf = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    solver=solver,
                    max_iter=1000,
                    random_state=42
                )
            
            try:
                clf.fit(X, y)
            except Exception as e:
                raise RuntimeError(
                    f"linear probe fit failed for layer {layer}: {e}"
                ) from e

            # The classifier weights point toward the positive class
            direction = clf.coef_[0]  # [hidden_dim]

            train_score = clf.score(X, y)
            model_scores[layer] = float(train_score)

            # Check weight sparsity (relevant for L1 regularization)
            non_zero_weights = np.count_nonzero(direction)
            sparsity_ratio = 1.0 - (non_zero_weights / len(direction))

            logger.info(
                f"Layer {layer}: accuracy {train_score:.4f}, sparsity "
                f"{sparsity_ratio:.3f} ({non_zero_weights}/{len(direction)} "
                f"non-zero weights)"
            )

            if non_zero_weights == 0:
                logger.warning(
                    f"Layer {layer}: all weights are zero; consider "
                    f"adjusting the regularization parameters."
                )

            if normalize:
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    logger.warning(f"Layer {layer}: cannot normalize a zero vector")

            directions[layer] = direction.astype(np.float32)

        metadata = {
            "normalize": normalize,
            "token_pos": token_pos,
            # The effective penalty; "none" stands for penalty=None
            "regularization": "none" if penalty is None else penalty,
            "C": C,
            "standardize": standardize,
            "n_positive": len(positive_indices),
            "n_negative": len(negative_indices),
            "classification_scores": model_scores
        }
        
        return StatisticalControlVector(
            model_type=model_type,
            method="linear_probe",
            directions=directions,
            metadata=metadata
        ) 