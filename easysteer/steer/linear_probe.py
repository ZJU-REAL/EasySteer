"""
Linear Probe Extractor
"""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base_extractor import BaseExtractor
from .utils import StatisticalControlVector, derive_negative_indices

logger = logging.getLogger(__name__)


def _build_classifier(penalty, C):
    """Construct the LogisticRegression for the requested penalty.

    Args:
        penalty (str | None): Effective sklearn penalty: "l1", "l2",
            "elasticnet", or None for an unregularized fit.
        C (float): Inverse regularization strength.

    Returns:
        LogisticRegression: The configured (unfitted) classifier.
    """
    if penalty == "elasticnet":
        return LogisticRegression(
            penalty=penalty,
            C=C,
            l1_ratio=0.5,  # elasticnet l1/l2 mixing ratio
            solver="saga",
            max_iter=1000,
            random_state=42,
        )
    if penalty is None:
        return LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=1000,
        random_state=42,
    )


class LinearProbeExtractor(BaseExtractor):
    """Linear Probe method for control vector extraction"""

    method = "linear_probe"
    progress_desc = "Computing LinearProbe directions"

    @staticmethod
    def _direction(pos_rows, neg_rows, *, layer, penalty, C, standardize):
        """Logistic-regression weights separating positives from negatives.

        Args:
            pos_rows (np.ndarray): Positive activations, `(n_pos, dim)`.
            neg_rows (np.ndarray): Negative activations, `(n_neg, dim)`.
            layer (int): Layer key, used for logging and errors.
            penalty (str | None): Effective sklearn penalty; None means
                no regularization.
            C (float): Inverse regularization strength.
            standardize (bool): Standardize features before fitting.

        Returns:
            tuple[np.ndarray, dict]: The classifier weights and the
                training accuracy under the `"classification_scores"`
                key.

        Raises:
            RuntimeError: If the classifier fit fails.
        """
        features = np.vstack([pos_rows, neg_rows])
        labels = np.hstack([
            np.ones(len(pos_rows)),  # positive samples labeled 1
            np.zeros(len(neg_rows)),  # negative samples labeled 0
        ])

        if standardize:
            features = StandardScaler().fit_transform(features)

        clf = _build_classifier(penalty, C)
        try:
            clf.fit(features, labels)
        except Exception as e:
            raise RuntimeError(
                f"linear probe fit failed for layer {layer}: {e}"
            ) from e

        # The classifier weights point toward the positive class
        direction = clf.coef_[0]  # [hidden_dim]
        train_score = float(clf.score(features, labels))

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

        return direction, {"classification_scores": train_score}

    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
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

        Returns:
            StatisticalControlVector: The extracted control vector.
        """
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
            "none": None,
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

        return LinearProbeExtractor._extract_template(
            all_hidden_states,
            positive_indices,
            negative_indices,
            normalize=normalize,
            token_pos=token_pos,
            opts={"penalty": penalty, "C": C, "standardize": standardize},
            extra_metadata={
                # The effective penalty; "none" stands for penalty=None
                "regularization": "none" if penalty is None else penalty,
                "C": C,
                "standardize": standardize,
            },
        )
