# SPDX-License-Identifier: Apache-2.0
"""Extracted directions and their GGUF serialization."""

import dataclasses
import os
import warnings

import numpy as np


@dataclasses.dataclass
class StatisticalControlVector:
    """Statistical control vector with per-layer directions."""

    method: str
    directions: dict[int, np.ndarray]
    metadata: dict | None = None
    # Only echoed into the gguf "model_hint" field for repeng
    # compatibility; nothing in EasySteer consumes it.
    model_type: str = "unknown"

    def export_gguf(self, path: os.PathLike[str] | str) -> None:
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
