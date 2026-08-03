# SPDX-License-Identifier: Apache-2.0
"""Adapters from third-party checkpoint formats to steering payloads.

The engine only loads formats whose schema EasySteer itself defines
(its GGUF export; the moe_router JSON). Everything else is interpreted
here, client-side: each adapter is a small pure function that reads one
known checkpoint layout and returns a canonical payload for
``VectorSpec(data=...)``. If your file does not match one of these
layouts, load it yourself and construct the payload directly — the
payload classes are the contract, the adapters are only conveniences.

Example:
    >>> from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
    >>> import easysteer.vectors as vec
    >>>
    >>> spec = SteeringSpec(vectors=[VectorSpec(
    ...     data=vec.from_lm_steer("gpt2.pt"),
    ...     algorithm="lm_steer",
    ...     layers=[11],
    ...     scale=1.0,
    ...     apply=ApplySpec(phases=["prompt", "generation"]),
    ... )])
"""

import glob
import json
import os
import pickle
from typing import Any

from vllm.steer_vectors.payloads import (
    DirectionVector,
    LinearMap,
    LowRankProjector,
    ReftIntervention,
)

__all__ = [
    "from_control_vector",
    "from_gguf",
    "from_linear_transport",
    "from_lm_steer",
    "from_pt_direction",
    "from_pyreft",
]


def from_pt_direction(path: str, layers: list[int]) -> DirectionVector:
    """Payload from a bare direction tensor saved with ``torch.save``.

    The file holds one vector (tensor or numpy array); it is applied to
    each listed layer.
    """
    import numpy as np
    import torch

    if not layers:
        raise ValueError("layers must be non-empty")
    vector = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector)
    if not isinstance(vector, torch.Tensor):
        raise ValueError(
            f"{path} does not contain a tensor or numpy array: "
            f"{type(vector).__name__}"
        )
    return DirectionVector({layer: vector for layer in layers})


def from_control_vector(cv: Any) -> DirectionVector:
    """Payload from an easysteer ``StatisticalControlVector``.

    The no-disk path: extract with ``easysteer.steer`` and steer with
    the result directly, no GGUF round-trip.
    """
    if not getattr(cv, "directions", None):
        raise ValueError("control vector has no directions")
    return DirectionVector(dict(cv.directions))


def from_gguf(path: str) -> DirectionVector:
    """Payload from an EasySteer GGUF export (``direction.<layer>``)."""
    from easysteer.steer.utils import import_gguf

    return from_control_vector(import_gguf(path))


def from_pyreft(path: str) -> DirectionVector | ReftIntervention:
    """Payload from a pyreft checkpoint directory.

    Reads the single ``*.bin`` + config pair. A BiasIntervention-style
    state dict (one vector) becomes a :class:`DirectionVector` for the
    ``direct`` algorithm; a LoReFT state dict (rotation + learned
    source) becomes a :class:`ReftIntervention` for ``loreft``. The
    checkpoint's layer index is preserved.
    """
    import torch

    bin_path, layer = _find_pyreft_checkpoint(path)
    state = torch.load(bin_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"{bin_path} does not hold a state dict: {type(state)}")

    rotate, weight, bias = None, None, None
    for key, value in state.items():
        if "rotate_layer" in key:
            if "parametrizations.weight.original" in key or key.endswith(
                "rotate_layer"
            ):
                rotate = value
        elif "learned_source" in key:
            if key.endswith("weight") and "parametrizations" not in key:
                weight = value
            elif key.endswith("bias"):
                bias = value
    if rotate is not None and weight is not None:
        return ReftIntervention(
            rotate_layer=rotate,
            learned_source_weight=weight,
            learned_source_bias=bias,
            layer=layer,
        )

    # BiasIntervention-style: exactly one plausible direction tensor.
    if len(state) == 1:
        vector = next(iter(state.values()))
    elif "source_representation" in state:
        vector = state["source_representation"]
    elif "bias" in state:
        vector = state["bias"]
    elif "weight" in state:
        vector = state["weight"]
    else:
        raise ValueError(
            f"cannot identify the intervention tensor in {bin_path}; "
            f"keys: {sorted(state)}. Load the checkpoint yourself and "
            "construct a payload directly."
        )
    return DirectionVector({layer: vector})


def from_lm_steer(path: str, vector_index: int = 0) -> LowRankProjector:
    """Payload from an LM-Steer checkpoint (.pt).

    Handles the published ``gpt2.pt`` layout (a list whose second entry
    is the parameter dict). Multi-vector checkpoints stack steer
    vectors; ``vector_index`` selects one — explicitly, instead of the
    silent first-vector default the engine loader used to apply.
    """
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, list) and len(state) > 1:
        state = state[1]
    if not isinstance(state, dict) or not (
        "projector1" in state and "projector2" in state
    ):
        raise ValueError(f"projector matrices not found in {path}")
    p1, p2 = state["projector1"], state["projector2"]
    if p1.dim() > 2:
        if not 0 <= vector_index < p1.shape[0]:
            raise ValueError(
                f"vector_index {vector_index} out of range for a "
                f"{p1.shape[0]}-vector checkpoint"
            )
        p1, p2 = p1[vector_index], p2[vector_index]
    elif vector_index != 0:
        raise ValueError("vector_index given but the checkpoint holds one vector")
    return LowRankProjector(projector1=p1, projector2=p2)


def from_linear_transport(path: str) -> LinearMap:
    """Payload from a LinearTransport pickle (``A_`` weight, ``B_`` bias)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        weight, bias = data.get("A_"), data.get("B_")
    else:
        weight, bias = getattr(data, "A_", None), getattr(data, "B_", None)
    if weight is None:
        raise ValueError(
            f"weight matrix (A_) not found in {path} (type {type(data).__name__})"
        )
    return LinearMap(weight=weight, bias=bias)


def _find_pyreft_checkpoint(path: str) -> tuple[str, int]:
    """Locate the weight file and layer index of a pyreft directory."""
    if not os.path.isdir(path):
        raise ValueError(f"pyreft checkpoint path must be a directory: {path}")
    bin_files = glob.glob(os.path.join(path, "*.bin"))
    if len(bin_files) != 1:
        raise ValueError(
            f"expected exactly one .bin file in {path}, found {len(bin_files)}"
        )
    config_files = [
        os.path.join(path, f)
        for f in ("reft_config.json", "config.json")
        if os.path.exists(os.path.join(path, f))
    ]
    if len(config_files) != 1:
        raise ValueError(
            f"expected exactly one config file in {path}, found "
            f"{len(config_files)}"
        )
    with open(config_files[0]) as f:
        config = json.load(f)

    layer = None
    representations = config.get("representations") or []
    if representations:
        first = representations[0]
        if isinstance(first, dict):
            layer = first.get("layer")
        elif isinstance(first, list) and first:
            layer = first[0]
    if layer is None:
        name = os.path.basename(bin_files[0])
        if "intkey_layer_" in name:
            layer_str = name.split("intkey_layer_")[1].split("_")[0]
            if layer_str.isdigit():
                layer = int(layer_str)
    if layer is None:
        raise ValueError(
            f"could not determine the layer index from {config_files[0]} "
            f"or the checkpoint filename"
        )
    return bin_files[0], int(layer)
