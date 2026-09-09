import enum
from .model import ReftModel


class ReftType(str, enum.Enum):
    """ReFT adapter identifiers."""

    LOREFT = "LOREFT"
    NLOREFT = "NOREFT"


class TaskType(str, enum.Enum):
    """Text classification and causal language modeling tasks."""

    SEQ_CLS = "SEQ_CLS"
    CAUSAL_LM = "CAUSAL_LM"


def get_reft_model(model, reft_config, set_device=True, disable_model_grads=True):
    """Wrap a model with the configured interventions and training settings."""
    reft_model = ReftModel(reft_config, model)
    if set_device:
        reft_model.set_device(model.device)
    if disable_model_grads:
        reft_model.disable_model_gradients()    
    return reft_model
