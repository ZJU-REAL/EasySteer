# SPDX-License-Identifier: Apache-2.0
"""Adapters from third-party checkpoint formats to steering payloads.

Adapters read known checkpoint layouts and return payloads for
``VectorSpec(data=...)``. For other layouts, construct a payload directly.
The engine itself loads EasySteer's GGUF and moe_router JSON formats.

Example:
    >>> from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
    >>> import easysteer.vectors as vec
    >>>
    >>> spec = SteeringSpec(vectors=[VectorSpec(
    ...     data=vec.from_lm_steer("gpt2.pt"),
    ...     algorithm="lm_steer",
    ...     layers=[11],
    ...     scale=1.0,
    ...     apply=ApplySpec(prompt="all", generation="all"),
    ... )])
"""

import base64
import glob
import json
import os
import pickle
from typing import Any

from vllm.model_hooks.steering.payloads import (
    DirectionVector,
    LinearMap,
    LowRankProjector,
    Payload,
    ReftIntervention,
)

__all__ = [
    "from_control_vector",
    "from_gguf",
    "from_linear_transport",
    "from_lm_steer",
    "from_pt_direction",
    "from_pyreft",
    "load",
    "to_json_payload",
]


def to_json_payload(payload: Payload) -> dict[str, Any]:
    """Encode a canonical payload for JSON files or OpenAI HTTP requests.

    Tensor bytes become base64 strings; shapes, metadata and the content
    hash remain identical to ``payload.to_wire()``. This runs in the
    producing EasySteer environment, so JSON consumers need no engine
    or tensor dependencies.
    """
    wire = payload.to_wire()
    return {
        **wire,
        "tensors": {
            name: {**tensor, "data": base64.b64encode(tensor["data"]).decode("ascii")}
            for name, tensor in wire["tensors"].items()
        },
    }


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
            f"{path} does not contain a tensor or numpy array: {type(vector).__name__}"
        )
    return DirectionVector({layer: vector for layer in layers})


def from_control_vector(cv: Any) -> DirectionVector:
    """Convert a StatisticalControlVector directly, without a GGUF round-trip."""
    if not getattr(cv, "directions", None):
        raise ValueError("control vector has no directions")
    return DirectionVector(dict(cv.directions))


def from_gguf(path: str) -> DirectionVector:
    """Payload from an EasySteer GGUF export (``direction.<layer>``)."""
    from vllm.model_hooks.steering.loading import load_file_payload

    return load_file_payload(path, format="gguf")


def load(path: str, *, format: str, **options) -> Payload:
    """Load an explicitly identified checkpoint schema into a canonical payload.

    Formats: ``gguf``, ``concept_pair`` (named h1/h2 GGUF directory),
    ``moe_router`` (JSON), ``pt_direction``, ``pyreft``, ``lm_steer`` and
    ``linear_transport``. Options go to the chosen adapter; for example
    ``layers=[10]`` for pt_direction or ``vector_index=1`` for lm_steer.
    A suffix such as .pt never selects or guesses a checkpoint schema.
    """
    if format in ("gguf", "concept_pair", "moe_router"):
        from vllm.model_hooks.steering.loading import load_file_payload

        return load_file_payload(path, format=format, **options)
    adapters = {
        "pt_direction": from_pt_direction,
        "pyreft": from_pyreft,
        "lm_steer": from_lm_steer,
        "linear_transport": from_linear_transport,
    }
    if format not in adapters:
        raise ValueError(f"Unknown checkpoint format: {format!r}")
    return adapters[format](path, **options)


def from_pyreft(path: str) -> DirectionVector | ReftIntervention:
    """Load one supported ReFT checkpoint without changing its target component.

    The checkpoint must identify a BiasIntervention or LoreftIntervention on
    ``block_output`` with ``unit="pos"`` and one unit. Bias becomes a
    ``DirectionVector`` for ``direct``; LoReFT becomes a ``ReftIntervention``
    for standard linear ``loreft``. Explicit nonlinear activation settings,
    other components and intervention types raise ValueError, even when their
    tensors happen to have the same width as hidden states. Older checkpoints
    without activation metadata are interpreted as standard linear LoReFT.

    The payload preserves the layer and weights. Token selection belongs to
    ``VectorSpec.apply``: use the checkpoint config's ``easysteer_training.apply``
    when present, or explicitly select the intended positions for older files.
    """
    import torch

    bin_path, layer, intervention = _find_pyreft_checkpoint(path)
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{bin_path} does not hold a state dict: {type(state)}")
    if intervention == "bias":
        if set(state) != {"bias"}:
            raise ValueError(
                f"{bin_path}: BiasIntervention requires exactly one bias tensor"
            )
        return DirectionVector({layer: state["bias"]})

    # Published PyReFT files use bare keys; earlier EasySteer exports used
    # the module-prefixed learned-source keys.
    layouts = (
        ("rotate_layer", "weight", "bias"),
        ("rotate_layer", "learned_source.weight", "learned_source.bias"),
    )
    for rotation, weight, bias in layouts:
        if set(state) == {rotation, weight, bias}:
            return ReftIntervention(
                rotate_layer=state[rotation],
                learned_source_weight=state[weight],
                learned_source_bias=state[bias],
                layer=layer,
            )
    raise ValueError(f"{bin_path}: unsupported LoReFT tensor keys: {sorted(state)}")


def from_lm_steer(path: str, vector_index: int = 0) -> LowRankProjector:
    """Payload from an LM-Steer checkpoint (.pt).

    Handles the published ``gpt2.pt`` layout (a list whose second entry
    is the parameter dict). Multi-vector checkpoints stack steer
    vectors; ``vector_index`` selects one and defaults to the first.
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


def _find_pyreft_checkpoint(path: str) -> tuple[str, int, str]:
    """Validate the serialized ReFT target before reading its weight tensor."""
    if not os.path.isdir(path):
        raise ValueError(f"pyreft checkpoint path must be a directory: {path}")
    bin_files = glob.glob(os.path.join(path, "*.bin"))
    if len(bin_files) != 1:
        raise ValueError(
            f"expected exactly one .bin file in {path}, found {len(bin_files)}"
        )
    config_files = [
        os.path.join(path, name)
        for name in ("reft_config.json", "config.json")
        if os.path.exists(os.path.join(path, name))
    ]
    if len(config_files) != 1:
        raise ValueError(
            f"expected exactly one config file in {path}, found {len(config_files)}"
        )
    with open(config_files[0]) as handle:
        config = json.load(handle)

    training = config.get("easysteer_training") or {}
    for act_fn in (config.get("act_fn"), training.get("act_fn")):
        if act_fn not in (None, "linear"):
            raise ValueError(
                f"Unsupported pyreft activation {act_fn!r}; "
                "vLLM ReFT conversion requires standard linear LoReFT"
            )

    representations = config.get("representations")
    if not isinstance(representations, list) or len(representations) != 1:
        raise ValueError("pyreft adapter requires exactly one representation")
    representation = representations[0]
    if isinstance(representation, list):
        fields = (
            "layer",
            "component",
            "unit",
            "max_number_of_units",
            "low_rank_dimension",
            "intervention_type",
            "intervention",
            "subspace_partition",
            "group_key",
            "intervention_link_key",
            "moe_key",
            "source_representation",
            "hidden_source_representation",
            "latent_dim",
        )
        if not 4 <= len(representation) <= len(fields):
            raise ValueError("pyreft representation does not describe its target units")
        representation = dict(zip(fields, representation))
    if not isinstance(representation, dict):
        raise TypeError("pyreft representation must be a dict or serialized field list")
    target = (
        representation.get("component"),
        representation.get("unit"),
        representation.get("max_number_of_units"),
    )
    if target != ("block_output", "pos", 1):
        raise ValueError(
            f"Unsupported pyreft target {target!r}; vLLM ReFT conversion requires "
            "component='block_output', unit='pos', max_number_of_units=1"
        )
    allowed = {
        "layer",
        "component",
        "unit",
        "max_number_of_units",
        "low_rank_dimension",
    }
    extra = {
        key: value
        for key, value in representation.items()
        if key not in allowed and value is not None
    }
    if extra:
        raise ValueError(f"Unsupported pyreft representation settings: {sorted(extra)}")
    layer = representation.get("layer")
    if type(layer) is not int or layer < 0:
        raise ValueError("pyreft representation requires a non-negative integer layer")

    types = config.get("intervention_types")
    if not isinstance(types, list) or len(types) != 1 or not isinstance(types[0], str):
        raise ValueError("pyreft config must identify exactly one intervention type")
    type_name = types[0].removeprefix("<class '").removesuffix("'>")
    supported = {
        "easysteer.reft.pyreft.reft.algorithms.bias.BiasIntervention": "bias",
        "easysteer.reft.pyreft.reft.algorithms.loreft.LoreftIntervention": "loreft",
        "pyreft.interventions.LoreftIntervention": "loreft",
    }
    if type_name not in supported:
        raise ValueError(f"Unsupported pyreft intervention type: {type_name!r}")
    return bin_files[0], layer, supported[type_name]
