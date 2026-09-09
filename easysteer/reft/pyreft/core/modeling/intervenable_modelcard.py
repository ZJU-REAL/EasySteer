"""Resolve component mappings without importing unrelated model families.

The registration set is explicit. A familiar-looking layer name or class
name in another Python module does not enable support for an unknown model.
"""

from functools import lru_cache
from importlib import import_module

from .model_profiles import ModelProfile, decoder_profile


# Qualified class name -> (mapping module, mapping prefix). Only this model's
# mapping module is imported when it is first used.
_PROFILE_SOURCES = {
    "transformers.models.gpt2.modeling_gpt2.GPT2Model": (
        "gpt2.modelings_intervenable_gpt2",
        "gpt2",
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel": (
        "gpt2.modelings_intervenable_gpt2",
        "gpt2_lm",
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification": (
        "gpt2.modelings_intervenable_gpt2",
        "gpt2_classifier",
    ),
    "transformers.models.llama.modeling_llama.LlamaModel": (
        "llama.modelings_intervenable_llama",
        "llama",
    ),
    "transformers.models.llama.modeling_llama.LlamaForCausalLM": (
        "llama.modelings_intervenable_llama",
        "llama_lm",
    ),
    "transformers.models.llama.modeling_llama.LlamaForSequenceClassification": (
        "llama.modelings_intervenable_llama",
        "llama_classifier",
    ),
    "transformers.models.llava.modeling_llava.LlavaForConditionalGeneration": (
        "llava.modelings_intervenable_llava",
        "llava",
    ),
    "transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoModel": (
        "gpt_neo.modelings_intervenable_gpt_neo",
        "gpt_neo",
    ),
    "transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM": (
        "gpt_neo.modelings_intervenable_gpt_neo",
        "gpt_neo_lm",
    ),
    "transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXModel": (
        "gpt_neox.modelings_intervenable_gpt_neox",
        "gpt_neox",
    ),
    "transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM": (
        "gpt_neox.modelings_intervenable_gpt_neox",
        "gpt_neox_lm",
    ),
    "transformers.models.mistral.modeling_mistral.MistralModel": (
        "mistral.modellings_intervenable_mistral",
        "mistral",
    ),
    "transformers.models.mistral.modeling_mistral.MistralForCausalLM": (
        "mistral.modellings_intervenable_mistral",
        "mistral_lm",
    ),
    "transformers.models.gemma.modeling_gemma.GemmaModel": (
        "gemma.modelings_intervenable_gemma",
        "gemma",
    ),
    "transformers.models.gemma.modeling_gemma.GemmaForCausalLM": (
        "gemma.modelings_intervenable_gemma",
        "gemma_lm",
    ),
    "transformers.models.gemma.modeling_gemma.GemmaForSequenceClassification": (
        "gemma.modelings_intervenable_gemma",
        "gemma_classifier",
    ),
    "transformers.models.gemma2.modeling_gemma2.Gemma2Model": (
        "gemma2.modelings_intervenable_gemma2",
        "gemma2",
    ),
    "transformers.models.gemma2.modeling_gemma2.Gemma2ForCausalLM": (
        "gemma2.modelings_intervenable_gemma2",
        "gemma2_lm",
    ),
    "transformers.models.olmo.modeling_olmo.OlmoModel": (
        "olmo.modelings_intervenable_olmo",
        "olmo",
    ),
    "transformers.models.olmo.modeling_olmo.OlmoForCausalLM": (
        "olmo.modelings_intervenable_olmo",
        "olmo_lm",
    ),
    "transformers.models.esm.modeling_esm.EsmModel": (
        "esm.modelings_intervenable_esm",
        "esm",
    ),
    "transformers.models.esm.modeling_esm.EsmForMaskedLM": (
        "esm.modelings_intervenable_esm",
        "esm_mlm",
    ),
    "transformers.models.blip.modeling_blip.BlipForQuestionAnswering": (
        "blip.modelings_intervenable_blip",
        "blip",
    ),
    "transformers.models.blip.modeling_blip.BlipForImageTextRetrieval": (
        "blip.modelings_intervenable_blip_itm",
        "blip_itm",
    ),
    f"{__package__}.mlp.modelings_mlp.MLPModel": (
        "mlp.modelings_intervenable_mlp",
        "mlp",
    ),
    f"{__package__}.mlp.modelings_mlp.MLPForClassification": (
        "mlp.modelings_intervenable_mlp",
        "mlp_classifier",
    ),
    f"{__package__}.gru.modelings_gru.GRUModel": (
        "gru.modelings_intervenable_gru",
        "gru",
    ),
    f"{__package__}.gru.modelings_gru.GRULMHeadModel": (
        "gru.modelings_intervenable_gru",
        "gru_lm",
    ),
    f"{__package__}.gru.modelings_gru.GRUForClassification": (
        "gru.modelings_intervenable_gru",
        "gru_classifier",
    ),
    f"{__package__}.backpack_gpt2.modelings_backpack_gpt2.BackpackGPT2LMHeadModel": (
        "backpack_gpt2.modelings_intervenable_backpack_gpt2",
        "backpack_gpt2_lm",
    ),
    "transformers.models.qwen2.modeling_qwen2.Qwen2Model": (
        "qwen2.modelings_intervenable_qwen2",
        "qwen2",
    ),
    "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM": (
        "qwen2.modelings_intervenable_qwen2",
        "qwen2_lm",
    ),
    "transformers.models.qwen2.modeling_qwen2.Qwen2ForSequenceClassification": (
        "qwen2.modelings_intervenable_qwen2",
        "qwen2_classifier",
    ),
    f"{__package__}.blip.modelings_blip.BlipWrapper": (
        "blip.modelings_intervenable_blip",
        "blip_wrapper",
    ),
    f"{__package__}.blip.modelings_blip_itm.BlipITMWrapper": (
        "blip.modelings_intervenable_blip",
        "blip_wrapper",
    ),
}

_SHARED_DECODERS = frozenset({"qwen2", "llama", "mistral", "gemma2"})


def model_family(model_type):
    """The registered family of an exact model type, or None."""
    key = f"{model_type.__module__}.{model_type.__qualname__}"
    source = _PROFILE_SOURCES.get(key)
    return source[0].split(".", 1)[0] if source is not None else None


@lru_cache(maxsize=None)
def get_model_profile(model_type) -> ModelProfile | None:
    key = f"{model_type.__module__}.{model_type.__qualname__}"
    source = _PROFILE_SOURCES.get(key)
    if source is None:
        return None
    module_name, mapping_name = source
    family = module_name.split(".", 1)[0]
    if family in _SHARED_DECODERS:
        return decoder_profile(prefix="" if mapping_name == family else "model.")
    module = import_module("." + module_name, package=__package__)
    dimensions = getattr(module, mapping_name + "_type_to_dimension_mapping")
    if key == f"{__package__}.blip.modelings_blip_itm.BlipITMWrapper":
        # Retain the existing wrapper's distinct dimension metadata. Its hook
        # paths are the common BLIP encoder paths registered above.
        itm = import_module(
            ".blip.modelings_intervenable_blip_itm", package=__package__
        )
        dimensions = itm.blip_itm_wrapper_type_to_dimension_mapping
    return ModelProfile(
        getattr(module, mapping_name + "_type_to_module_mapping"),
        dimensions,
    )
