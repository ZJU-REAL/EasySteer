"""
Utility classes and functions for steering methods
Contains the StatisticalControlVector class and shared helpers
"""

import dataclasses
import os
import warnings
from pathlib import Path

import gguf
import numpy as np
import torch

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StatisticalControlVector:
    """Statistical control vector with multi-layer directions"""
    model_type: str
    method: str
    directions: dict[int, np.ndarray]
    metadata: dict = None

    def export_gguf(self, path: os.PathLike[str] | str):
        """
        Export a trained StatisticalControlVector to a llama.cpp .gguf file.
        Compatible with repeng format.
        """
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
                    # Handle nested dictionaries like explained_variance
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            writer.add_float32(f"{arch}.{key}.{subkey}", float(subvalue))
        
        for layer in self.directions.keys():
            writer.add_tensor(f"direction.{layer}", self.directions[layer])
        
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "StatisticalControlVector":
        """Import a StatisticalControlVector from a .gguf file"""
        reader = gguf.GGUFReader(path)

        archf = reader.get_field("general.architecture")
        if not archf or not len(archf.parts):
            warnings.warn(".gguf file missing architecture field")
        else:
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "controlvector":
                warnings.warn(
                    f".gguf file with architecture {arch!r} does not appear to be a control vector!"
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
        
        # Extract metadata
        for field_name, field in reader.fields.items():
            if field_name.startswith("controlvector.") and not field_name.endswith((".model_hint", ".method", ".layer_count")):
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

        return cls(model_type=model_hint, method=method, directions=directions, metadata=metadata)


def derive_negative_indices(n_samples, positive_indices):
    """Derive negative sample indices from the shared convention.

    Every extractor uses one rule when ``negative_indices`` is None:
    the negatives are every sample index not in ``positive_indices``,
    in ascending sample order. ``positive_indices`` is never modified
    or rebound.

    Args:
        n_samples (int): Total number of samples.
        positive_indices (list[int]): Indices of positive samples.

    Returns:
        list[int]: Indices of negative samples, in ascending order.
    """
    positive = set(positive_indices)
    return [i for i in range(n_samples) if i not in positive]


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
            Never modified or rebound by this function.
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
            array. Layer keys are the TRUE layer ids for CaptureResult
            input and positional indices for nested-list input.
    """
    # CaptureResult (easysteer.hidden_states) is accepted directly; it
    # converts to the nested [sample][layer][token] shape with exact,
    # label-driven per-sample rows, and its TRUE layer ids key the
    # output dicts (legacy nested input keeps positional keys).
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

    positive_hiddens = {layer: [] for layer in layer_keys}
    negative_hiddens = {layer: [] for layer in layer_keys}

    def extract_token_from_sequence(token_sequence, pos):
        """Extract the token at the requested position from a sequence"""
        if isinstance(pos, int):
            return token_sequence[pos]
        elif pos == "first":
            return token_sequence[0]
        elif pos == "last":
            return token_sequence[-1]
        elif pos == "mean":
            tokens = np.stack([t.cpu().float().numpy() if torch.is_tensor(t) else t for t in token_sequence])
            return np.mean(tokens, axis=0)
        elif pos == "max":
            # Token with the largest L2 norm
            norms = []
            tokens = []
            for t in token_sequence:
                if torch.is_tensor(t):
                    t = t.cpu().float().numpy()
                tokens.append(t)
                norms.append(np.linalg.norm(t))
            max_idx = np.argmax(norms)
            return tokens[max_idx]
        elif pos == "min":
            # Token with the smallest L2 norm
            norms = []
            tokens = []
            for t in token_sequence:
                if torch.is_tensor(t):
                    t = t.cpu().float().numpy()
                tokens.append(t)
                norms.append(np.linalg.norm(t))
            min_idx = np.argmin(norms)
            return tokens[min_idx]
        else:
            raise ValueError(f"Unsupported token_pos: {pos}")

    for sample_idx in positive_indices:
        sample_hiddens = all_hidden_states[sample_idx]
        for layer_pos, layer_key in enumerate(layer_keys):
            token_hidden = extract_token_from_sequence(
                sample_hiddens[layer_pos], token_pos
            )
            if torch.is_tensor(token_hidden):
                token_hidden = token_hidden.cpu().float().numpy()
            positive_hiddens[layer_key].append(token_hidden)

    if negative_indices:
        for sample_idx in negative_indices:
            sample_hiddens = all_hidden_states[sample_idx]
            for layer_pos, layer_key in enumerate(layer_keys):
                token_hidden = extract_token_from_sequence(
                    sample_hiddens[layer_pos], token_pos
                )
                if torch.is_tensor(token_hidden):
                    token_hidden = token_hidden.cpu().float().numpy()
                negative_hiddens[layer_key].append(token_hidden)

    positive_hiddens = {k: np.vstack(v) for k, v in positive_hiddens.items()}
    if negative_indices and any(negative_hiddens.values()):
        negative_hiddens = {k: np.vstack(v) for k, v in negative_hiddens.items()}
    else:
        negative_hiddens = {}

    return positive_hiddens, negative_hiddens



