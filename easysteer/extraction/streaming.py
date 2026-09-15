# SPDX-License-Identifier: Apache-2.0
"""Labelled capture extraction with bounded numerical working storage."""

from copy import deepcopy
from numbers import Integral

import numpy as np

from ._utils import l2_normalize
from .accumulators import (
    DEFAULT_WORKING_BYTES,
    check_memory_budget,
    validate_memory_budget,
)
from .result import StatisticalControlVector

_TOKEN_CHUNK_SIZE = 32


def _is_capture(value):
    return hasattr(value, "sample_rows") and hasattr(value, "layer_ids")


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        return value.numpy()
    return value


def _capture_widths(capture):
    widths = {}
    for layer in capture.layer_ids:
        if isinstance(layer, bool) or not isinstance(layer, Integral) or layer < 0:
            raise ValueError("capture layer IDs must be non-negative integers")
        shape = capture.layers[layer].shape
        if len(shape) != 2 or shape[1] < 1:
            raise ValueError(
                f"layer {layer}: activations must have shape (rows, width)"
            )
        widths[layer] = shape[1]
    if not widths:
        raise ValueError("capture contains no layers")
    return widths


def _next_label(labels, sample):
    try:
        value = next(labels)
    except StopIteration as exc:
        raise ValueError(f"labels ended before sample {sample}") from exc
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, (bool, Integral, np.bool_)) or value not in (0, 1):
        raise ValueError(f"sample {sample}: labels must be Boolean or integer 0/1")
    return bool(value)


def _finish_labels(labels):
    sentinel = object()
    if next(labels, sentinel) is not sentinel:
        raise ValueError("labels contains more entries than captured samples")


def _reduce_row(capture, sample, layer, token_pos):
    position = {"first": 0, "last": -1}.get(token_pos, token_pos)
    try:
        if isinstance(position, Integral):
            row = _to_numpy(capture.token(sample, layer, int(position)))
        else:
            if hasattr(capture, "iter_sample_rows"):
                chunks = capture.iter_sample_rows(
                    sample, layer, chunk_size=_TOKEN_CHUNK_SIZE
                )
            else:
                # Compatibility for external CaptureResult-like containers.
                rows = capture.sample_rows(sample, layer)
                chunks = (
                    rows[i : i + _TOKEN_CHUNK_SIZE]
                    for i in range(0, len(rows), _TOKEN_CHUNK_SIZE)
                )
            row, best_norm, count = None, None, 0
            for values in chunks:
                values = np.asarray(_to_numpy(values), dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"sample {sample}, layer {layer}: nonfinite token row"
                    )
                count += len(values)
                if position == "mean":
                    if row is None:
                        row = values.sum(axis=0)
                    else:
                        row += values.sum(axis=0)
                else:
                    norms = np.linalg.norm(values, axis=1)
                    index = np.argmax(norms) if position == "max" else np.argmin(norms)
                    norm = norms[index]
                    if best_norm is None or (
                        norm > best_norm if position == "max" else norm < best_norm
                    ):
                        row, best_norm = values[index].copy(), norm
            if count == 0:
                raise IndexError
            if position == "mean":
                row /= count
    except IndexError as exc:
        raise ValueError(
            f"sample {sample}, layer {layer}: token_pos={token_pos!r} "
            "does not select a captured row"
        ) from exc
    row = np.asarray(row, dtype=np.float64)
    if row.ndim != 1 or not np.isfinite(row).all():
        raise ValueError(
            f"sample {sample}, layer {layer}: expected a finite activation row"
        )
    return row


def _has_per_prompt_selection(capture):
    return any(
        selection is not None
        for selection in (getattr(capture, "per_prompt_selections", None) or ())
    )


def _provenance(capture):
    return (
        getattr(capture, "component", None),
        getattr(capture, "model", None),
        deepcopy(getattr(capture, "selection", None)),
    )


class _PooledCapture:
    """Expose one lazily pooled row per sample to materialized estimators."""

    def __init__(self, capture, token_pos):
        self.capture = capture
        self.token_pos = token_pos

    def __getattr__(self, name):
        return getattr(self.capture, name)

    def __len__(self):
        return len(self.capture)

    def token(self, sample, layer, position=0):
        if position not in (0, -1):
            raise IndexError("pooled captures contain one row per sample")
        return _reduce_row(self.capture, sample, layer, self.token_pos)

    def sample_rows(self, sample, layer):
        return self.token(sample, layer)[None, :]


def _materialized(capture, labels, method, token_pos, normalize, budget, options):
    from .api import extract_statistical_control_vector

    if method not in {"pca", "lat", "linear_probe", "iti"}:
        raise ValueError(f"Unknown extraction method: {method!r}")
    positive, negative = [], []
    for sample in range(len(capture)):
        (positive if _next_label(labels, sample) else negative).append(sample)
    _finish_labels(labels)
    if not capture.layer_ids or not len(capture):
        raise ValueError("capture contains no samples or layers")
    widths = _capture_widths(capture)
    width = max(widths.values())
    n_rows = len(capture)
    validation = options.get("validation_hidden_states")
    if method == "iti" and validation is not None:
        if not _is_capture(validation):
            raise TypeError(
                "bounded ITI extraction requires a validation CaptureResult"
            )
        validation_widths = _capture_widths(validation)
        if widths != validation_widths:
            raise ValueError(
                "training and validation captures must have the same layer widths"
            )
        n_rows += len(validation)
    # Preflight reduced rows, estimator copies, preprocessing and decomposition
    # scratch conservatively. This is an allocation estimate, not an RSS cap on
    # Python, sklearn or the underlying BLAS library.
    required = 8 * (20 * n_rows * width + 8 * min(n_rows, width) ** 2)
    required += sum(capture.layers[layer].shape[-1] * 4 for layer in capture.layer_ids)
    required += 32 * _TOKEN_CHUNK_SIZE * width
    check_memory_budget(required, budget, operation=f"{method} extraction")
    if method == "iti":
        if normalize:
            raise ValueError(
                "ITI preserves projection scale and requires normalize=False"
            )
    else:
        options["normalize"] = normalize
    if method == "iti" and "validation_labels" in options:
        if validation is None:
            raise ValueError("validation_labels requires validation_hidden_states")
        if {
            "validation_positive_indices",
            "validation_negative_indices",
        } & options.keys():
            raise ValueError(
                "provide validation_labels or validation indices, not both"
            )
        validation_labels = iter(options.pop("validation_labels"))
        validation_positive, validation_negative = [], []
        for sample in range(len(validation)):
            target = (
                validation_positive
                if _next_label(validation_labels, sample)
                else validation_negative
            )
            target.append(sample)
        _finish_labels(validation_labels)
        options["validation_positive_indices"] = validation_positive
        options["validation_negative_indices"] = validation_negative
    if method == "iti" and _is_capture(validation):
        options["validation_hidden_states"] = _PooledCapture(validation, token_pos)
    result = extract_statistical_control_vector(
        method,
        _PooledCapture(capture, token_pos),
        positive,
        negative,
        token_pos=0,
        **options,
    )
    result.metadata["token_pos"] = token_pos
    result.metadata["capture_selection"] = deepcopy(getattr(capture, "selection", None))
    result.metadata["capture_has_per_prompt_selections"] = _has_per_prompt_selection(
        capture
    )
    return result


class _IncrementalLayer:
    def __init__(self, width, batch_size):
        from sklearn.decomposition import IncrementalPCA

        self.rows = np.empty((batch_size, width), dtype=np.float64)
        self.count = 0
        self.pca = IncrementalPCA(n_components=1, batch_size=batch_size)

    def update(self, row):
        self.rows[self.count] = row
        self.count += 1
        if self.count == len(self.rows):
            self.flush()

    def flush(self):
        if self.count:
            self.pca.partial_fit(self.rows[: self.count])
            # sklearn keeps slices of Vt/S. Copy the retained rank-one state so
            # those views cannot hold a whole batch-by-width SVD allocation.
            for name in (
                "components_",
                "singular_values_",
                "explained_variance_",
                "explained_variance_ratio_",
            ):
                setattr(self.pca, name, getattr(self.pca, name).copy())
            self.count = 0


def extract(
    captures,
    labels,
    *,
    method: str = "diffmean",
    token_pos: int | str = -1,
    normalize: bool | None = None,
    max_working_bytes: int = DEFAULT_WORKING_BYTES,
    pca_batch_size: int = 32,
    correct_direction: bool = True,
    **options,
) -> StatisticalControlVector:
    """Extract a vector from a capture or an ordered stream of capture batches.

    Labels are Boolean or integer 0/1, one per sample in capture order. Every
    sample contributes one row per layer, including when ``token_pos="mean"``
    pools a variable number of captured tokens. ``diffmean`` retains only class
    sums. ``incremental_pca`` fits the positive rows with bounded IncrementalPCA
    buffers; it is approximate and may depend on sample order and batch size.

    ``max_working_bytes`` bounds estimated numerical allocations made by this
    function, excluding input captures and runtime/library overhead. Capture
    batches need their own capture budget. PCA/LAT/probe/ITI also accept a single
    materialized CaptureResult, with a conservative allocation preflight; their
    method-specific options are forwarded to the corresponding extractor.
    ITI accepts ``validation_hidden_states`` and a matching ``validation_labels``
    iterable, or the existing explicit validation index arguments.
    """
    budget = validate_memory_budget(max_working_bytes)
    if normalize is None:
        normalize = method != "iti"
    if isinstance(token_pos, bool) or not (
        isinstance(token_pos, Integral)
        or token_pos in ("first", "last", "mean", "max", "min")
    ):
        raise ValueError(f"Unsupported token_pos: {token_pos!r}")
    if not isinstance(normalize, bool) or not isinstance(correct_direction, bool):
        raise TypeError("normalize and correct_direction must be Boolean")
    labels = iter(labels)
    single = _is_capture(captures)
    if method not in {"diffmean", "incremental_pca"}:
        if not single:
            raise ValueError(
                "Capture streams support method='diffmean' or 'incremental_pca'. "
                "Other methods require a single materialized CaptureResult."
            )
        if method in {"pca", "lat"}:
            options["correct_direction"] = correct_direction
        return _materialized(
            captures, labels, method, token_pos, normalize, budget, options
        )
    if options:
        raise ValueError(f"Unknown option(s) {sorted(options)} for method {method!r}")
    if (
        isinstance(pca_batch_size, bool)
        or not isinstance(pca_batch_size, Integral)
        or pca_batch_size < 2
    ):
        raise ValueError("pca_batch_size must be an integer >= 2")

    buffer_size = (
        min(int(pca_batch_size), max(2, len(captures)))
        if single
        else int(pca_batch_size)
    )
    batches = iter((captures,)) if single else iter(captures)
    widths, provenance = None, None
    sums, pcas = {}, {}
    counts = [0, 0]
    has_per_prompt_selections = False
    for capture in batches:
        if not _is_capture(capture):
            raise TypeError("captures must contain CaptureResult batches")
        if not len(capture):
            continue
        current = _capture_widths(capture)
        has_per_prompt_selections |= _has_per_prompt_selection(capture)
        if widths is None:
            widths, provenance = current, _provenance(capture)
            total_width, largest = sum(widths.values()), max(widths.values())
            required = 32 * total_width + 32 * _TOKEN_CHUNK_SIZE * largest
            if method == "incremental_pca":
                n_svd = buffer_size + 2
                required += 8 * (buffer_size + 12) * total_width
                required += 8 * (8 * n_svd * largest + 8 * min(n_svd, largest) ** 2)
            check_memory_budget(required, budget, operation=f"streaming {method}")
            sums = {
                layer: np.zeros((2, width), dtype=np.float64)
                for layer, width in widths.items()
            }
            if method == "incremental_pca":
                pcas = {
                    layer: _IncrementalLayer(width, buffer_size)
                    for layer, width in widths.items()
                }
        elif current != widths or _provenance(capture) != provenance:
            raise ValueError(
                "capture batches must have identical layers, widths, component, model and selection"
            )
        for sample in range(len(capture)):
            positive = _next_label(labels, sum(counts))
            for layer in widths:
                row = _reduce_row(capture, sample, layer, token_pos)
                if row.shape != (widths[layer],):
                    raise ValueError(f"layer {layer}: row width does not match capture")
                sums[layer][int(positive)] += row
                if method == "incremental_pca" and positive:
                    pcas[layer].update(row)
            counts[int(positive)] += 1
            del row
        # Drop the consumed batch before asking its producer for the next one.
        del capture
    _finish_labels(labels)
    if widths is None:
        raise ValueError("capture stream contains no samples")
    if counts[1] == 0 or (method == "diffmean" and counts[0] == 0):
        raise ValueError(
            f"{method} requires positive samples"
            + (" and negative samples" if method == "diffmean" else "")
        )
    if method == "incremental_pca" and counts[1] < 2:
        raise ValueError("incremental_pca requires at least two positive samples")

    directions, variance = {}, {}
    for layer in widths:
        gap = sums[layer][1] / counts[1]
        if counts[0]:
            gap = gap - sums[layer][0] / counts[0]
        if method == "diffmean":
            direction = gap
        else:
            pcas[layer].flush()
            pca = pcas[layer].pca
            if not np.isfinite(pca.explained_variance_ratio_[0]):
                raise ValueError(f"layer {layer}: PCA requires nonzero variance")
            direction = pca.components_[0].copy()
            variance[layer] = float(pca.explained_variance_ratio_[0])
            if correct_direction and counts[0] and float(direction @ gap) < 0:
                direction = -direction
        if normalize:
            direction = l2_normalize(direction)
        directions[layer] = direction.astype(np.float32)
    component, model, selection = provenance
    metadata = {
        "normalize": normalize,
        "n_positive": counts[1],
        "n_negative": counts[0],
        "token_pos": token_pos,
        "streaming": True,
        "capture_selection": selection,
        "capture_has_per_prompt_selections": has_per_prompt_selections,
    }
    if method == "incremental_pca":
        metadata.update(
            approximate=True,
            pca_batch_size=int(pca_batch_size),
            explained_variance=variance,
            correct_direction=correct_direction,
        )
    return StatisticalControlVector(
        method=method,
        directions=directions,
        metadata=metadata,
        component=component,
        model_type=model or "unknown",
    )
