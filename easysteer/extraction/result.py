# SPDX-License-Identifier: Apache-2.0
"""Extracted directions, target-aware steering and GGUF serialization."""

import dataclasses
import json
import os
import warnings

import numpy as np


def _encode_metadata(value):
    """Preserve integer layer keys, tuples and scalars in a JSON field."""
    if isinstance(value, dict):
        return {
            "mapping": [
                [_encode_metadata(k), _encode_metadata(v)] for k, v in value.items()
            ]
        }
    if isinstance(value, tuple):
        return {"tuple": [_encode_metadata(item) for item in value]}
    if isinstance(value, list):
        return [_encode_metadata(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported extraction metadata type: {type(value).__name__}")


def _decode_metadata(value):
    if isinstance(value, dict):
        if set(value) == {"mapping"}:
            return {
                _decode_metadata(k): _decode_metadata(v) for k, v in value["mapping"]
            }
        if set(value) == {"tuple"}:
            return tuple(_decode_metadata(item) for item in value["tuple"])
        raise ValueError("invalid EasySteer GGUF metadata")
    if isinstance(value, list):
        return [_decode_metadata(item) for item in value]
    return value


def _field_value(field, gguf):
    """Read GGUF values using value types, never tensor quantization types."""
    if hasattr(field, "contents"):
        return field.contents()
    if field.types == [gguf.GGUFValueType.STRING]:
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    if len(field.data) == 1:
        return field.parts[field.data[0]].reshape(-1)[0].item()
    return [part.tolist() for part in (field.parts[i] for i in field.data)]


@dataclasses.dataclass
class StatisticalControlVector:
    """Per-layer directions with their capture component and analysis metadata.

    ``component`` is unknown for legacy unlabelled inputs. Such vectors require
    an explicit component when converted to a steering spec. Capture selection
    describes the analysis data; inference selection is always passed explicitly.
    """

    method: str
    directions: dict[int, np.ndarray]
    metadata: dict | None = None
    model_type: str = "unknown"
    component: str | None = None

    def to_spec(self, *, apply, scale: float = 1.0, component: str | None = None):
        """Create a complete vLLM spec without guessing the target or positions.

        Hidden-state directions use ``direct`` and attention-head directions use
        ``attention_add``. Router-logit directions are analysis results: the
        ``moe_router`` algorithm requires explicit expert selections and modes,
        so it cannot consume an additive direction vector.
        """
        if component is not None and self.component not in (None, component):
            raise ValueError("component conflicts with the captured vector target")
        target = component or self.component
        algorithms = {"hidden_states": "direct", "attention_heads": "attention_add"}
        if target not in algorithms:
            raise ValueError(
                f"Cannot convert component {target!r} to an additive steering spec. "
                "Legacy vectors require an explicit hidden_states or attention_heads "
                "component; router_logits require a moe_router expert-selection payload."
            )
        from vllm.steer_vectors import SteeringSpec, VectorSpec

        from easysteer.vectors import from_control_vector

        return SteeringSpec(
            vectors=[
                VectorSpec(
                    data=from_control_vector(self),
                    algorithm=algorithms[target],
                    layers=sorted(self.directions),
                    scale=scale,
                    apply=apply,
                )
            ]
        )

    def export_gguf(self, path: os.PathLike[str] | str) -> None:
        """Export repeng-compatible directions and lossless native metadata."""
        import gguf

        # Validate metadata before opening the destination file.
        native = json.dumps(
            {
                "version": 1,
                "component": self.component,
                "metadata": _encode_metadata(self.metadata),
            },
            allow_nan=False,
            separators=(",", ":"),
        )
        arch = "controlvector"
        writer = gguf.GGUFWriter(path, arch)
        try:
            writer.add_string(f"{arch}.model_hint", self.model_type)
            writer.add_string(f"{arch}.method", self.method)
            writer.add_uint32(f"{arch}.layer_count", len(self.directions))
            writer.add_string(f"{arch}.easysteer", native)
            for layer, direction in self.directions.items():
                writer.add_tensor(
                    f"direction.{layer}", np.asarray(direction, dtype=np.float32)
                )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "StatisticalControlVector":
        """Read native metadata and older flattened numeric/string metadata."""
        import gguf

        reader = gguf.GGUFReader(path)
        archf = reader.get_field("general.architecture")
        if archf is None:
            warnings.warn(".gguf file missing architecture field", stacklevel=2)
        else:
            arch = _field_value(archf, gguf)
            if arch != "controlvector":
                warnings.warn(
                    f".gguf architecture {arch!r} is not controlvector", stacklevel=2
                )

        modelf = reader.get_field("controlvector.model_hint")
        if modelf is None:
            raise ValueError(".gguf file missing controlvector.model_hint field")
        methodf = reader.get_field("controlvector.method")
        nativef = reader.get_field("controlvector.easysteer")
        component = None
        if nativef is not None:
            native = json.loads(_field_value(nativef, gguf))
            if native.get("version") != 1:
                raise ValueError("unsupported EasySteer GGUF metadata version")
            metadata = _decode_metadata(native["metadata"])
            component = native.get("component")
        else:
            metadata = {}
            reserved = {"model_hint", "method", "layer_count"}
            for name, field in reader.fields.items():
                if name.startswith("controlvector."):
                    key = name.removeprefix("controlvector.")
                    if key not in reserved:
                        metadata[key] = _field_value(field, gguf)

        directions = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.removeprefix("direction."))
                if layer < 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    f"invalid direction field name: {tensor.name}"
                ) from exc
            directions[layer] = np.array(tensor.data, copy=True)
        return cls(
            model_type=_field_value(modelf, gguf),
            method=_field_value(methodf, gguf) if methodf is not None else "unknown",
            directions=directions,
            metadata=metadata,
            component=component,
        )
