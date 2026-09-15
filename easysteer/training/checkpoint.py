# SPDX-License-Identifier: Apache-2.0
"""Versioned training metadata paired with a canonical inference payload."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
CHECKPOINT_NAME = "steering_adapter.json"


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "loreft"
    component: str = "hidden_states"
    layer: int = 8
    rank: int | None = 4
    prompt_template: str = PROMPT_TEMPLATE

    def __post_init__(self):
        if self.algorithm not in {"direct", "loreft"}:
            raise ValueError("training algorithm must be 'direct' or 'loreft'")
        if self.component != "hidden_states":
            raise ValueError("direct and loreft training require 'hidden_states'")
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if self.algorithm == "direct":
            object.__setattr__(self, "rank", None)
        elif type(self.rank) is not int or self.rank < 1:
            raise ValueError("LoReFT rank must be a positive integer")
        if not isinstance(self.prompt_template, str):
            raise TypeError("prompt_template must be a string")
        try:
            self.prompt_template % "instruction"
        except (TypeError, ValueError) as exc:
            raise ValueError("prompt_template must contain one %s placeholder") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "component": self.component,
            "layer": self.layer,
            "rank": self.rank,
            "prompt_template": self.prompt_template,
            "apply": {"prompt_positions": [-1]},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        fields = {"algorithm", "component", "layer", "rank", "prompt_template", "apply"}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("checkpoint config has missing or unknown fields")
        if data["apply"] != {"prompt_positions": [-1]}:
            raise ValueError("training checkpoints require the last prompt position")
        return cls(**{key: value for key, value in data.items() if key != "apply"})


@dataclass(frozen=True)
class SteeringCheckpoint:
    config: TrainingConfig
    payload: Any

    def __post_init__(self):
        from vllm.model_hooks.steering.payloads import DirectionVector, ReftIntervention

        if self.config.algorithm == "direct":
            valid = isinstance(self.payload, DirectionVector) and set(
                self.payload.layers
            ) == {self.config.layer}
        else:
            import numpy as np

            valid = (
                isinstance(self.payload, ReftIntervention)
                and self.payload.layer == self.config.layer
                and self.payload.rotate_layer.shape[1] == self.config.rank
                and self.payload.learned_source_bias is not None
                and np.allclose(
                    self.payload.rotate_layer.T @ self.payload.rotate_layer,
                    np.eye(self.config.rank),
                    atol=1e-5,
                    rtol=1e-5,
                )
            )
        if not valid:
            raise ValueError(
                "checkpoint payload does not match its training config; "
                "LoReFT requires a bias and an orthonormal rotation"
            )

    def to_spec(self):
        from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

        return SteeringSpec(
            vectors=[
                VectorSpec(
                    data=self.payload,
                    algorithm=self.config.algorithm,
                    scale=1.0,
                    apply=ApplySpec(prompt_positions=[-1]),
                )
            ]
        )

    def save(self, path: str | Path) -> Path:
        from easysteer.vectors import to_json_payload

        target = Path(path)
        if target.suffix != ".json":
            target = target / CHECKPOINT_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "config": self.config.to_dict(),
            "payload": to_json_payload(self.payload),
        }
        # Replace atomically so readers cannot observe half of a checkpoint.
        import tempfile

        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=target.parent, suffix=".json", delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(data, handle, indent=2, allow_nan=False)
                handle.write("\n")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target


def load_checkpoint(path: str | Path) -> SteeringCheckpoint:
    """Load native weights and their exact inference target and token selection."""
    from vllm.model_hooks.steering.payloads import from_wire

    target = Path(path)
    if target.is_dir():
        target = target / CHECKPOINT_NAME
    data = json.loads(target.read_text())
    if not isinstance(data, dict) or set(data) != {"version", "config", "payload"}:
        raise ValueError("invalid native steering checkpoint")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("unsupported steering checkpoint version")
    return SteeringCheckpoint(
        TrainingConfig.from_dict(data["config"]), from_wire(data["payload"])
    )
