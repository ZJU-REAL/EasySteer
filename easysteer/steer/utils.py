"""Control-vector serialization, token selection, and shared extraction math."""

import dataclasses
import logging
import os
import warnings

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StatisticalControlVector:
    """Statistical control vector with per-layer directions."""

    method: str
    directions: dict[int, np.ndarray]
    metadata: dict = None
    # Only echoed into the gguf "model_hint" field for repeng
    # compatibility; nothing in EasySteer consumes it.
    model_type: str = "unknown"

    def export_gguf(self, path: os.PathLike[str] | str):
        """Export directions and metadata to a repeng-compatible GGUF file."""
        import gguf

        arch = "controlvector"
        writer = gguf.GGUFWriter(path, arch)
        writer.add_string(f"{arch}.model_hint", self.model_type)
        writer.add_string(f"{arch}.method", self.method)
        writer.add_uint32(f"{arch}.layer_count", len(self.directions))

        if self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, (int, float)):
                    writer.add_float32(f"{arch}.{key}", float(value))
                elif isinstance(value, str):
                    writer.add_string(f"{arch}.{key}", value)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            writer.add_float32(
                                f"{arch}.{key}.{subkey}", float(subvalue)
                            )

        for layer in self.directions.keys():
            writer.add_tensor(f"direction.{layer}", self.directions[layer])

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "StatisticalControlVector":
        """Import a StatisticalControlVector from a GGUF file."""
        import gguf

        reader = gguf.GGUFReader(path)

        archf = reader.get_field("general.architecture")
        if not archf or not len(archf.parts):
            warnings.warn(".gguf file missing architecture field")
        else:
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "controlvector":
                warnings.warn(
                    f".gguf file with architecture {arch!r} does not "
                    f"appear to be a control vector!"
                )

        modelf = reader.get_field("controlvector.model_hint")
        if not modelf or not len(modelf.parts):
            raise ValueError(".gguf file missing controlvector.model_hint field")
        model_hint = str(bytes(modelf.parts[-1]), encoding="utf-8")

        methodf = reader.get_field("controlvector.method")
        method = "unknown"
        if methodf and len(methodf.parts):
            method = str(bytes(methodf.parts[-1]), encoding="utf-8")

        directions = {}
        metadata = {}

        skipped_suffixes = (".model_hint", ".method", ".layer_count")
        for field_name, field in reader.fields.items():
            if field_name.startswith("controlvector.") and not (
                field_name.endswith(skipped_suffixes)
            ):
                key = field_name.replace("controlvector.", "")
                if field.types == [gguf.GGMLQuantizationType.F32]:
                    metadata[key] = float(field.parts[0])
                elif field.types == [gguf.GGMLQuantizationType.I32]:
                    metadata[key] = int(field.parts[0])

        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f".gguf file has invalid direction field name: {tensor.name}"
                )
            directions[layer] = tensor.data

        return cls(
            model_type=model_hint,
            method=method,
            directions=directions,
            metadata=metadata,
        )


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
            easysteer.hidden_states.
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
    layer_keys = None
    if hasattr(all_hidden_states, "to_nested"):
        layer_keys = list(all_hidden_states.layer_ids)
        all_hidden_states = all_hidden_states.to_nested()
    if negative_indices is None:
        negative_indices = derive_negative_indices(
            len(all_hidden_states), positive_indices
        )

    n_layers = len(all_hidden_states[0])
    if layer_keys is None:
        layer_keys = list(range(n_layers))

    def collect(indices):
        """Gather one token row per sample for each layer key."""
        hiddens = {layer: [] for layer in layer_keys}
        for sample_idx in indices:
            sample_hiddens = all_hidden_states[sample_idx]
            for layer_pos, layer_key in enumerate(layer_keys):
                token_hidden = extract_token_from_sequence(
                    sample_hiddens[layer_pos], token_pos
                )
                hiddens[layer_key].append(_to_numpy(token_hidden))
        return hiddens

    positive_hiddens = collect(positive_indices)
    negative_hiddens = collect(negative_indices or [])

    positive_hiddens = {k: np.vstack(v) for k, v in positive_hiddens.items()}
    if negative_indices and any(negative_hiddens.values()):
        negative_hiddens = {k: np.vstack(v) for k, v in negative_hiddens.items()}
    else:
        negative_hiddens = {}

    return positive_hiddens, negative_hiddens
