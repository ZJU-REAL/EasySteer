"""
Unified Interface for Steering Methods
"""

import inspect

from .utils import StatisticalControlVector
from .diffmean import DiffMeanExtractor
from .pca import PCAExtractor
from .lat import LATExtractor
from .linear_probe import LinearProbeExtractor

_EXTRACTORS = {
    "diffmean": DiffMeanExtractor,
    "pca": PCAExtractor,
    "lat": LATExtractor,
    "linear_probe": LinearProbeExtractor,
}

_COMMON_PARAMS = ("all_hidden_states", "positive_indices", "negative_indices")


def _accepted_options(extract_fn):
    """List the keyword options an extractor's extract() accepts.

    The shared positional data arguments and the catch-all ``**kwargs``
    are excluded, so the result names exactly the options a caller may
    pass through the unified interface.

    Args:
        extract_fn (Callable): The extractor's extract function.

    Returns:
        list[str]: Accepted option names, in signature order.
    """
    signature = inspect.signature(extract_fn)
    return [
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY)
        and name not in _COMMON_PARAMS
    ]


def _dispatch(method, all_hidden_states, positive_indices, negative_indices,
              kwargs):
    """Validate options against the target extractor and dispatch.

    Every public extraction function funnels through here: the method
    name resolves to an extractor, and each keyword option is checked
    against that extractor's actual signature so a typo raises instead
    of being silently ignored.

    Args:
        method (str): Extraction method name.
        all_hidden_states (list | CaptureResult): Hidden states input.
        positive_indices (list[int]): Indices of positive samples.
        negative_indices (list[int] | None): Indices of negative
            samples, or None to derive them.
        kwargs (dict): Method-specific options to validate.

    Returns:
        StatisticalControlVector: The extracted control vector.

    Raises:
        ValueError: If ``method`` is unknown, or any option in
            ``kwargs`` is not accepted by the target extractor.
    """
    if method not in _EXTRACTORS:
        raise ValueError(
            f"Unsupported method: {method!r}. Supported methods: "
            f"{list(_EXTRACTORS)}"
        )
    extract_fn = _EXTRACTORS[method].extract
    accepted = _accepted_options(extract_fn)
    unknown = sorted(set(kwargs) - set(accepted))
    if unknown:
        raise ValueError(
            f"Unknown option(s) {unknown} for method {method!r}. "
            f"Accepted options: {accepted}"
        )
    return extract_fn(
        all_hidden_states=all_hidden_states,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        **kwargs,
    )


def extract_statistical_control_vector(
    method: str,
    all_hidden_states,
    positive_indices,
    negative_indices=None,
    **kwargs
) -> StatisticalControlVector:
    """Unified control vector extraction interface.

    Args:
        method (str): Method name; one of "diffmean", "pca", "lat",
            "linear_probe".
        all_hidden_states (list | CaptureResult): Nested
            `[sample][layer][token]` hidden states, or a CaptureResult
            from easysteer.hidden_states.
        positive_indices (list[int]): Indices of positive samples.
        negative_indices (list[int] | None): Indices of negative
            samples. If None, every sample index not in
            ``positive_indices`` becomes a negative, in ascending
            sample order.
        **kwargs (Any): Method-specific options. Unknown options raise
            ValueError naming the accepted ones for ``method``.

    Returns:
        StatisticalControlVector: The extracted control vector.
    """
    return _dispatch(
        method, all_hidden_states, positive_indices, negative_indices, kwargs
    )


def extract_diffmean_control_vector(
    all_hidden_states, positive_indices, negative_indices=None, **kwargs
) -> StatisticalControlVector:
    """Extract a DiffMean control vector.

    Args:
        all_hidden_states (list | CaptureResult): Nested
            `[sample][layer][token]` hidden states, or a CaptureResult
            from easysteer.hidden_states.
        positive_indices (list[int]): Indices of positive samples.
        negative_indices (list[int] | None): Indices of negative
            samples. If None, every sample index not in
            ``positive_indices`` becomes a negative, in ascending
            sample order.
        **kwargs (Any): Options accepted by DiffMeanExtractor:
            `model_type` (str), `normalize` (bool, default True),
            `token_pos` (int | str, default -1). Unknown options raise
            ValueError.

    Returns:
        StatisticalControlVector: The DiffMean control vector.
    """
    return _dispatch(
        "diffmean", all_hidden_states, positive_indices, negative_indices,
        kwargs
    )


def extract_pca_control_vector(
    all_hidden_states, positive_indices, negative_indices=None, **kwargs
) -> StatisticalControlVector:
    """Extract a PCA control vector.

    Args:
        all_hidden_states (list | CaptureResult): Nested
            `[sample][layer][token]` hidden states, or a CaptureResult
            from easysteer.hidden_states.
        positive_indices (list[int]): Indices of positive samples.
        negative_indices (list[int] | None): Indices of negative
            samples. If None, every sample index not in
            ``positive_indices`` becomes a negative, in ascending
            sample order.
        **kwargs (Any): Options accepted by PCAExtractor:
            `method` (str, default "standard") selects the PCA variant
            ("standard" uses only positive samples, "diff" runs PCA
            over positive/negative differences, "center" over
            pair-centered samples); `correct_direction` (bool, default
            True) flips the vector if needed so it points from negative
            toward positive samples; `n_components` (int, must be 1);
            `normalize` (bool, default True); `token_pos` (int | str,
            default -1, the last token); `model_type` (str). Unknown
            options raise ValueError.

    Returns:
        StatisticalControlVector: The PCA control vector.

    Examples:
        >>> # Plain PCA over positive samples only
        >>> pca_vector = extract_pca_control_vector(
        ...     all_hidden_states, positive_indices,
        ...     method="standard"
        ... )
        >>>
        >>> # PCA over pair differences with direction correction
        >>> pca_diff_vector = extract_pca_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     method="diff", correct_direction=True
        ... )
        >>>
        >>> # PCA over pair differences without direction correction
        >>> pca_diff_no_correct = extract_pca_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     method="diff", correct_direction=False
        ... )
    """
    return _dispatch(
        "pca", all_hidden_states, positive_indices, negative_indices, kwargs
    )


def extract_lat_control_vector(
    all_hidden_states, positive_indices, negative_indices=None, **kwargs
) -> StatisticalControlVector:
    """Extract a LAT control vector.

    Args:
        all_hidden_states (list | CaptureResult): Nested
            `[sample][layer][token]` hidden states, or a CaptureResult
            from easysteer.hidden_states.
        positive_indices (list[int]): Indices of positive samples.
        negative_indices (list[int] | None): Indices of negative
            samples. If None and `use_positive_only` is False, every
            sample index not in ``positive_indices`` becomes a
            negative, in ascending sample order.
        **kwargs (Any): Options accepted by LATExtractor:
            `use_positive_only` (bool, default True) uses only the
            positive samples; `correct_direction` (bool, default True)
            flips the vector if needed so it points from negative
            toward positive samples; `n_components` (int, default 1);
            `normalize` (bool, default True); `token_pos` (int | str,
            default -1, the last token); `model_type` (str). Unknown
            options raise ValueError.

    Returns:
        StatisticalControlVector: The LAT control vector.

    Examples:
        >>> # Positive samples only (traditional LAT)
        >>> lat_vector = extract_lat_control_vector(
        ...     all_hidden_states, positive_indices,
        ...     use_positive_only=True
        ... )
        >>>
        >>> # Positive and negative samples with direction correction
        >>> lat_mixed_vector = extract_lat_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     use_positive_only=False, correct_direction=True
        ... )
    """
    return _dispatch(
        "lat", all_hidden_states, positive_indices, negative_indices, kwargs
    )


def extract_linear_probe_control_vector(
    all_hidden_states, positive_indices, negative_indices=None, **kwargs
) -> StatisticalControlVector:
    """Extract a Linear Probe control vector.

    Args:
        all_hidden_states (list | CaptureResult): Nested
            `[sample][layer][token]` hidden states, or a CaptureResult
            from easysteer.hidden_states.
        positive_indices (list[int]): Indices of positive samples.
        negative_indices (list[int] | None): Indices of negative
            samples. If None, every sample index not in
            ``positive_indices`` becomes a negative, in ascending
            sample order.
        **kwargs (Any): Options accepted by LinearProbeExtractor:
            `regularization` (str, default "l2") is one of "l1", "l2",
            "elasticnet", "none" — unknown values raise ValueError;
            `C` (float, default 1.0) is the inverse regularization
            strength (for L2, C=1.0 is usually fine; for L1, prefer
            C=10.0 or larger to avoid excessive sparsification; with
            "none" C is ignored); `standardize` (bool, default True);
            `normalize` (bool, default True); `token_pos` (int | str,
            default -1, the last token); `model_type` (str). Unknown
            options raise ValueError.

    Returns:
        StatisticalControlVector: The Linear Probe control vector.

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
    return _dispatch(
        "linear_probe", all_hidden_states, positive_indices,
        negative_indices, kwargs
    )
