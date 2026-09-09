"""Component layouts shared by explicitly supported decoder families."""

from dataclasses import dataclass

from .constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK, split_head_and_permute


@dataclass(frozen=True)
class ModelProfile:
    modules: dict
    dimensions: dict


def _head_dimension(config):
    if getattr(config, "head_dim", None) is not None:
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


def _query_dimension(config):
    return config.num_attention_heads * _head_dimension(config)


def _kv_dimension(config):
    return config.num_key_value_heads * _head_dimension(config)


def decoder_profile(*, prefix: str) -> ModelProfile:
    """The common Qwen2/Llama/Mistral/Gemma2 layout, with explicit geometry.

    This describes known architectures; it does not recognize an arbitrary
    model by its module names. KV heads remain distinct from query heads,
    and an explicit head_dim takes precedence over hidden_size / num_heads.
    """
    layer = prefix + "layers[%s]"
    attention = layer + ".self_attn"
    modules = {
        "block_input": (layer, CONST_INPUT_HOOK),
        "block_output": (layer, CONST_OUTPUT_HOOK),
        "mlp_activation": (layer + ".mlp.act_fn", CONST_OUTPUT_HOOK),
        "mlp_output": (layer + ".mlp", CONST_OUTPUT_HOOK),
        "mlp_input": (layer + ".mlp", CONST_INPUT_HOOK),
        "attention_value_output": (attention + ".o_proj", CONST_INPUT_HOOK),
        "head_attention_value_output": (
            attention + ".o_proj",
            CONST_INPUT_HOOK,
            (split_head_and_permute, "n_head"),
        ),
        "attention_output": (attention, CONST_OUTPUT_HOOK),
        "attention_input": (attention, CONST_INPUT_HOOK),
    }
    for component, projection, head_count in (
        ("query", "q_proj", "n_head"),
        ("key", "k_proj", "n_kv_head"),
        ("value", "v_proj", "n_kv_head"),
    ):
        modules[component + "_output"] = (
            attention + "." + projection,
            CONST_OUTPUT_HOOK,
        )
        modules["head_" + component + "_output"] = (
            attention + "." + projection,
            CONST_OUTPUT_HOOK,
            (split_head_and_permute, head_count),
        )
    dimensions = {component: ("hidden_size",) for component in modules}
    dimensions.update(
        {
            "n_head": ("num_attention_heads",),
            "n_kv_head": ("num_key_value_heads",),
            "mlp_activation": ("intermediate_size",),
            "attention_value_output": (_query_dimension,),
            "query_output": (_query_dimension,),
            "key_output": (_kv_dimension,),
            "value_output": (_kv_dimension,),
        }
    )
    for component in modules:
        if component.startswith("head_"):
            dimensions[component] = (_head_dimension,)
    return ModelProfile(modules, dimensions)
