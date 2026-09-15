# SPDX-License-Identifier: Apache-2.0
"""Shared numerical operations and extraction metadata."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def l2_normalize(v):
    """Scale a vector to unit L2 norm, leaving zero vectors unchanged."""
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v


def correct_sign(component, pos_rows, neg_rows):
    """Point ``component`` from the negative toward the positive samples.

    Args:
        component (np.ndarray): Candidate direction, shape `(dim,)`.
        pos_rows (np.ndarray): Positive activations, `(n_pos, dim)`.
        neg_rows (np.ndarray): Negative activations, `(n_neg, dim)`.

    Returns:
        np.ndarray: ``component``, or its negation if it pointed from
            the positive toward the negative samples.
    """
    vec_norm = np.linalg.norm(component)
    if vec_norm <= 1e-6:  # A near-zero vector has no meaningful sign.
        return component
    proj_pos = (pos_rows @ component) / vec_norm
    proj_neg = (neg_rows @ component) / vec_norm
    if np.mean(proj_pos) < np.mean(proj_neg):
        logger.info("Direction corrected (flipped)")
        return -component
    return component


def _metadata(*, normalize, n_positive, n_negative, **extra):
    """Combine sample counts and normalization with method-specific metadata."""
    metadata = {
        "normalize": normalize,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }
    metadata.update(extra)
    return metadata
