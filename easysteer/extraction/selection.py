# SPDX-License-Identifier: Apache-2.0
"""Select labelled samples and reduce captured token rows."""

import numpy as np


def derive_negative_indices(n_samples, positive_indices):
    """Return sample indices outside positive_indices in ascending order."""
    positive = set(positive_indices)
    return [i for i in range(n_samples) if i not in positive]


def _to_numpy(value):
    """Detach tensor-like rows and convert them through CPU float32."""
    return value.detach().cpu().float().numpy() if hasattr(value, "detach") else value


def _tokens_to_numpy(token_sequence):
    """Convert tensor-like token rows to CPU NumPy arrays."""
    return [_to_numpy(token) for token in token_sequence]


def _extreme_norm_token(argfn):
    """Build a token reducer using np.argmax or np.argmin of the L2 norms."""

    def reducer(token_sequence):
        tokens = _tokens_to_numpy(token_sequence)
        norms = [np.linalg.norm(t) for t in tokens]
        return tokens[argfn(norms)]

    return reducer


_TOKEN_REDUCERS = {
    "first": lambda seq: seq[0],
    "last": lambda seq: seq[-1],
    "mean": lambda seq: np.mean(np.stack(_tokens_to_numpy(seq)), axis=0),
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
    if isinstance(pos, int):
        return token_sequence[pos]
    reducer = _TOKEN_REDUCERS.get(pos)
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
    all_hidden_states, positive_indices, negative_indices=None, token_pos=-1
):
    """Yield one layer's selected positive/negative rows at a time.

    For CaptureResult, integer positions index the sample's captured rows in
    sequence order. Select a scalar row before materializing other layers or
    positions; reducers needing multiple rows only read the current layer.
    """
    if negative_indices is None:
        negative_indices = derive_negative_indices(
            len(all_hidden_states), positive_indices
        )
    captured = hasattr(all_hidden_states, "sample_rows")
    layers = (
        all_hidden_states.layer_ids if captured else range(len(all_hidden_states[0]))
    )
    position = {"first": 0, "last": -1}.get(token_pos, token_pos)

    def collect(indices, layer):
        rows = []
        for sample in indices:
            if captured and isinstance(position, int):
                row = all_hidden_states.token(sample, layer, position)
            else:
                sequence = (
                    all_hidden_states.sample_rows(sample, layer)
                    if captured
                    else all_hidden_states[sample][layer]
                )
                row = extract_token_from_sequence(sequence, position)
            rows.append(_to_numpy(row))
        return np.vstack(rows)

    for layer in layers:
        positive = collect(positive_indices, layer)
        negative = collect(negative_indices, layer) if negative_indices else None
        yield layer, positive, negative
