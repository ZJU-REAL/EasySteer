# Core intervention framework - extracted from pyvene
from .base import IntervenableModel, IntervenableConfig, IntervenableNdifModel, build_intervenable_model
from .base import RepresentationConfig
from .interventions import *
from .utils import *

# Core tensor utilities do not import model families.
from .modeling.common import *

# Preserve the model-construction entry points without importing every family
# (and its optional Transformers dependencies) when ReFT is imported.
_FACTORY_MODULES = {
    "create_gpt2": ".modeling.gpt2",
    "create_gpt2_lm": ".modeling.gpt2",
    "create_llama": ".modeling.llama",
    "create_blip": ".modeling.blip",
    "create_blip_itm": ".modeling.blip",
    "create_gpt_neo": ".modeling.gpt_neo",
    "create_gpt_neox": ".modeling.gpt_neox",
    "create_gru": ".modeling.gru",
    "create_gru_lm": ".modeling.gru",
    "create_gru_classifier": ".modeling.gru",
    "GRUConfig": ".modeling.gru",
    "create_llava": ".modeling.llava",
    "create_mlp_classifier": ".modeling.mlp",
    "create_backpack_gpt2": ".modeling.backpack_gpt2",
    "create_olmo": ".modeling.olmo",
}


def __getattr__(name):
    if name not in _FACTORY_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(_FACTORY_MODULES[name], __name__), name)
    globals()[name] = value
    return value
