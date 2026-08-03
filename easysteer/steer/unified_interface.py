"""
Unified interface for the control vector extraction (steering) methods.
"""

from .utils import StatisticalControlVector
from .diffmean import DiffMeanExtractor
from .pca import PCAExtractor
from .lat import LATExtractor
from .linear_probe import LinearProbeExtractor


def extract_statistical_control_vector(
    method: str,
    all_hidden_states,
    positive_indices,
    negative_indices=None,
    **kwargs
) -> StatisticalControlVector:
    """
    Unified interface for extracting a control vector.

    Args:
        method (str): Method name; one of "diffmean", "pca", "lat",
            or "linear_probe".
        all_hidden_states (list): Nested list of hidden states indexed by
            `[sample][layer][token]`.
        positive_indices (list): Indices of the positive samples.
        negative_indices (list): Indices of the negative samples.
        **kwargs (Any): Method-specific parameters.

    Returns:
        StatisticalControlVector: The extracted control vector.
    """
    method_map = {
        "diffmean": DiffMeanExtractor,
        "pca": PCAExtractor,
        "lat": LATExtractor,
        "linear_probe": LinearProbeExtractor,
    }
    
    if method not in method_map:
        supported_methods = list(method_map.keys())
        raise ValueError(f"不支持的方法: {method}。支持的方法: {supported_methods}")
    
    extractor_class = method_map[method]
    return extractor_class.extract(
        all_hidden_states=all_hidden_states,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        **kwargs
    )


def extract_diffmean_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """Extract a DiffMean control vector."""
    return DiffMeanExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_pca_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """
    Extract a PCA control vector.

    Args:
        all_hidden_states (list): Nested list of hidden states indexed by
            `[sample][layer][token]`.
        positive_indices (list): Indices of the positive samples.
        negative_indices (list): Indices of the negative samples.
        **kwargs (Any): Method-specific options:

            - use_positive_only (bool): If True (default), use only the
              positive samples (traditional PCA); if False, run PCA on the
              differences between positive and negative samples.
            - correct_direction (bool): Whether to correct the vector
              direction so that it points from the negative samples to the
              positive samples. Defaults to True.
            - n_components (int): Number of PCA components. Defaults to 1.
            - normalize (bool): Whether to normalize the vector. Defaults
              to True.
            - token_pos (int): Token position. Defaults to -1 (the last
              token).

    Returns:
        (StatisticalControlVector): The extracted PCA control vector.

    Examples:
        >>> # Use only the positive samples (traditional PCA)
        >>> pca_vector = extract_pca_control_vector(
        ...     all_hidden_states, positive_indices,
        ...     use_positive_only=True
        ... )
        >>>
        >>> # Run PCA on positive/negative differences, with direction
        >>> # correction enabled (the default)
        >>> pca_diff_vector = extract_pca_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     use_positive_only=False, correct_direction=True
        ... )
        >>>
        >>> # Run PCA on positive/negative differences, without direction
        >>> # correction
        >>> pca_diff_no_correct = extract_pca_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     use_positive_only=False, correct_direction=False
        ... )
    """
    return PCAExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_lat_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """
    Extract a LAT control vector.

    Args:
        all_hidden_states (list): Nested list of hidden states indexed by
            `[sample][layer][token]`.
        positive_indices (list): Indices of the positive samples.
        negative_indices (list): Indices of the negative samples.
        **kwargs (Any): Method-specific options:

            - use_positive_only (bool): Whether to use only the positive
              samples. Defaults to True.
            - correct_direction (bool): Whether to correct the vector
              direction so that it points from the negative samples to the
              positive samples. Defaults to True.
            - n_components (int): Number of PCA components. Defaults to 1.
            - normalize (bool): Whether to normalize the vector. Defaults
              to True.
            - token_pos (int): Token position. Defaults to -1 (the last
              token).

    Returns:
        (StatisticalControlVector): The extracted LAT control vector.

    Examples:
        >>> # Use only the positive samples (traditional LAT)
        >>> lat_vector = extract_lat_control_vector(
        ...     all_hidden_states, positive_indices,
        ...     use_positive_only=True
        ... )
        >>>
        >>> # Use positive and negative samples, with direction correction
        >>> # enabled (the default)
        >>> lat_mixed_vector = extract_lat_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     use_positive_only=False, correct_direction=True
        ... )
    """
    return LATExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_linear_probe_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """
    Extract a Linear Probe control vector.

    Args:
        all_hidden_states (list): Nested list of hidden states indexed by
            `[sample][layer][token]`.
        positive_indices (list): Indices of the positive samples.
        negative_indices (list): Indices of the negative samples.
        **kwargs (Any): Method-specific options:

            - regularization (str): Regularization type ("l1", "l2",
              "elasticnet", "none"). Defaults to "l2".
            - C (float): Inverse of the regularization strength. Defaults
              to 1.0. Note: C=1.0 is usually fine for L2 regularization;
              for L1, prefer C=10.0 or larger to avoid over-sparsification;
              without regularization the C parameter is ignored.
            - standardize (bool): Whether to standardize the features.
              Defaults to True.

    Returns:
        (StatisticalControlVector): The extracted Linear Probe control
            vector.

    Examples:
        >>> # L2 regularization (recommended)
        >>> linear_probe_vector = extract_linear_probe_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     model_type="qwen2.5", regularization="l2", C=1.0
        ... )
        >>>
        >>> # L1 regularization (feature selection)
        >>> linear_probe_l1 = extract_linear_probe_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     model_type="qwen2.5", regularization="l1", C=10.0
        ... )
    """
    return LinearProbeExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs) 