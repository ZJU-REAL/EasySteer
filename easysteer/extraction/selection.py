# SPDX-License-Identifier: Apache-2.0
"""Select labelled samples and reduce captured token rows."""

from numbers import Integral

import numpy as np


def _validate_indices(n_samples, indices, name):
    try:
        indices = list(indices)
    except TypeError as exc:
        raise ValueError(f"{name} must contain integer sample indices") from exc
    if any(
        isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < n_samples
        for i in indices
    ):
        raise ValueError(f"{name} must contain integer indices in [0, {n_samples})")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must contain unique sample indices")
    return [int(i) for i in indices]


def validate_sample_groups(
    n_samples,
    positive_indices,
    negative_indices=None,
    *,
    require_negative=True,
    derive_negative=True,
):
    """Validate disjoint sample groups and optionally derive the complement.

    Indices address samples within this input, not absolute capture token
    positions or the global sample indices of a streamed capture batch.
    """
    positive = _validate_indices(n_samples, positive_indices, "positive_indices")
    if not positive:
        raise ValueError("positive_indices must be nonempty")
    if negative_indices is None:
        negative_indices = (
            derive_negative_indices(n_samples, positive) if derive_negative else []
        )
    negative = _validate_indices(n_samples, negative_indices, "negative_indices")
    if set(positive) & set(negative):
        raise ValueError("positive and negative indices must be disjoint")
    if require_negative and not negative:
        raise ValueError("negative_indices must be nonempty")
    return positive, negative


def derive_negative_indices(n_samples, positive_indices):
    """Return sample indices outside positive_indices in ascending order."""
    positive = set(_validate_indices(n_samples, positive_indices, "positive_indices"))
    return [i for i in range(n_samples) if i not in positive]


def _to_numpy(value):
    """Detach tensor rows without narrowing NumPy-compatible capture dtypes."""
    if not hasattr(value, "detach"):
        return value
    value = value.detach().cpu()
    dtype = getattr(value, "dtype", None)
    if dtype is None or str(dtype) == "torch.bfloat16":
        # NumPy has no bfloat16. Keep the existing float conversion contract
        # for tensor-like objects that do not expose their storage dtype.
        value = value.float()
    return value.numpy()


def _extreme_norm_token(argfn):
    """Build a token reducer using np.argmax or np.argmin of the L2 norms."""

    def reducer(token_sequence):
        chosen, chosen_norm = None, None
        for token in token_sequence:
            token = _to_numpy(token)
            norm = np.linalg.norm(token)
            if chosen is None or argfn([chosen_norm, norm]) == 1:
                chosen, chosen_norm = token, norm
        return chosen

    return reducer


def _mean_token(token_sequence):
    """Pool rows without creating a second full token matrix."""
    if isinstance(token_sequence, np.ndarray):
        return token_sequence.mean(axis=0)
    total = None
    for token in token_sequence:
        row = np.asarray(_to_numpy(token), dtype=np.float64)
        if total is None:
            total = row.copy()
        else:
            total += row
    return total / len(token_sequence)


_TOKEN_REDUCERS = {
    "first": lambda seq: seq[0],
    "last": lambda seq: seq[-1],
    "mean": _mean_token,
    "max": _extreme_norm_token(np.argmax),
    "min": _extreme_norm_token(np.argmin),
}


def extract_token_from_sequence(token_sequence, pos):
    """Reduce a per-token sequence to one hidden-state row.

    Args:
        token_sequence (Sequence): Hidden states of one layer of one
            sample, as tensors or numpy arrays.
        pos (int | str): An int index (e.g. -1 for the last token), or
            one of "first", "last", "mean" (average over tokens),
            "max"/"min" (token with the largest/smallest L2 norm).

    Returns:
        np.ndarray | torch.Tensor: The selected or aggregated row.

    Raises:
        ValueError: If ``pos`` is neither an int nor a known reducer
            name.
    """
    if len(token_sequence) == 0:
        raise ValueError("no captured rows match the sample selection")
    if isinstance(pos, Integral) and not isinstance(pos, bool):
        if not -len(token_sequence) <= pos < len(token_sequence):
            raise ValueError(f"token_pos={pos} is outside the captured rows")
        return token_sequence[pos]
    reducer = _TOKEN_REDUCERS.get(pos) if isinstance(pos, str) else None
    if reducer is None:
        raise ValueError(f"Unsupported token_pos: {pos}")
    return reducer(token_sequence)


def extract_token_hiddens(
    all_hidden_states, positive_indices, negative_indices=None, token_pos=-1
) -> tuple[dict, dict]:
    """Extract hidden states of one token position per sample.

    Args:
        all_hidden_states (list | CaptureResult): Nested
            `[sample][layer][token]` hidden states, where each entry is
            a tensor or numpy array, or a CaptureResult from
            easysteer.capture.
        positive_indices (list[int]): Indices of positive samples.
        negative_indices (list[int] | None): Indices of negative
            samples. If None, every sample index not in
            ``positive_indices`` becomes a negative, in ascending
            sample order (the convention shared by all extractors).
        token_pos (int | str): Token position to extract:
            an int index (-1 selects the last token, the default),
            "first", "last", "mean" (average over tokens), "max" or
            "min" (token with the largest/smallest L2 norm).

    Returns:
        tuple[dict, dict]: `(positive_hiddens, negative_hiddens)`, each
            a dict mapping layer key to a `(n_samples, hidden_dim)`
            array. Layer keys are the model layer ids for CaptureResult
            input and positional indices for nested-list input.
    """
    positive_hiddens, negative_hiddens = {}, {}
    for layer, positive, negative in iter_token_hiddens(
        all_hidden_states, positive_indices, negative_indices, token_pos
    ):
        positive_hiddens[layer] = positive
        if negative is not None:
            negative_hiddens[layer] = negative
    return positive_hiddens, negative_hiddens


def iter_token_hiddens(
    all_hidden_states,
    positive_indices,
    negative_indices=None,
    token_pos=-1,
    *,
    layers=None,
):
    """Yield one layer's selected positive/negative rows at a time.

    For CaptureResult, integer positions index the sample's captured rows in
    sequence order. Select a scalar row before materializing other layers or
    positions; reducers needing multiple rows only read the current layer.
    """
    positive_indices, negative_indices = validate_sample_groups(
        len(all_hidden_states),
        positive_indices,
        negative_indices,
        require_negative=False,
    )
    captured = hasattr(all_hidden_states, "sample_rows")
    available_layers = (
        all_hidden_states.layer_ids if captured else range(len(all_hidden_states[0]))
    )
    layers = list(available_layers if layers is None else layers)
    if not layers or not set(layers) <= set(available_layers):
        raise ValueError("input must contain every requested capture layer")
    if not (
        isinstance(token_pos, Integral)
        and not isinstance(token_pos, bool)
        or isinstance(token_pos, str)
        and token_pos in _TOKEN_REDUCERS
    ):
        raise ValueError(f"Unsupported token_pos: {token_pos}")
    position = {"first": 0, "last": -1}.get(token_pos, token_pos)

    def collect(indices, layer):
        rows = []
        for sample in indices:
            try:
                if captured and isinstance(position, Integral):
                    row = all_hidden_states.token(sample, layer, position)
                else:
                    sequence = (
                        all_hidden_states.sample_rows(sample, layer)
                        if captured
                        else all_hidden_states[sample][layer]
                    )
                    row = extract_token_from_sequence(sequence, position)
            except (IndexError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"sample {sample}, layer {layer}: no captured row for "
                    f"token_pos={token_pos!r}; check the capture selection"
                ) from exc
            row = np.asarray(_to_numpy(row))
            if row.ndim != 1 or row.size == 0 or not np.isfinite(row).all():
                raise ValueError(
                    f"sample {sample}, layer {layer}: expected a finite feature vector"
                )
            if rows and row.shape != rows[0].shape:
                raise ValueError(f"layer {layer}: inconsistent activation widths")
            rows.append(row)
        return np.vstack(rows)

    for layer in layers:
        positive = collect(positive_indices, layer)
        negative = collect(negative_indices, layer) if negative_indices else None
        if negative is not None and positive.shape[1] != negative.shape[1]:
            raise ValueError(f"layer {layer}: inconsistent activation widths")
        yield layer, positive, negative
