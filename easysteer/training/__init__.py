# SPDX-License-Identifier: Apache-2.0
"""Train steering adapters on frozen Hugging Face causal language models.

Checkpoint loading needs only the canonical vLLM payload types. Torch and
Transformers are imported when model loading or training is requested.
"""

from typing import TYPE_CHECKING

from .checkpoint import (
    PROMPT_TEMPLATE,
    SteeringCheckpoint,
    TrainingConfig,
    load_checkpoint,
)

if TYPE_CHECKING:
    from .api import generate, load, train
    from .model import SteeringModel

__all__ = [
    "PROMPT_TEMPLATE",
    "SteeringCheckpoint",
    "SteeringModel",
    "TrainingConfig",
    "generate",
    "load",
    "load_checkpoint",
    "train",
]


def __getattr__(name):
    if name in {"generate", "load", "train"}:
        from . import api

        return getattr(api, name)
    if name == "SteeringModel":
        from .model import SteeringModel

        return SteeringModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
