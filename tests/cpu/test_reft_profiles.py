"""Tiny CPU models exercise hook paths and head geometry without checkpoints."""

from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

from easysteer.reft.pyreft.core.modeling.intervenable_modelcard import get_model_profile
from easysteer.reft.pyreft.core.modeling.modeling_utils import (
    get_dimension_by_component,
    get_module_hook,
    output_to_subcomponent,
)


@pytest.fixture(params=["qwen2", "llama", "mistral", "gemma2"])
def model(request):
    # Resolve only the family exercised by this case.
    from importlib import import_module

    names = {
        "qwen2": "Qwen2",
        "llama": "Llama",
        "mistral": "Mistral",
        "gemma2": "Gemma2",
    }
    name = names[request.param]
    hf = import_module("transformers")
    config = getattr(hf, name + "Config")(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        sliding_window=16,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return getattr(hf, name + "ForCausalLM")(config)


def test_supported_decoder_hooks_share_one_layout(model):
    profile = get_model_profile(type(model))
    assert len(profile.modules) == 15
    hook = get_module_hook(
        model, SimpleNamespace(component="block_output", layer=1, moe_key=None)
    )
    assert hook.__self__ is model.model.layers[1]
    hook = get_module_hook(
        model,
        SimpleNamespace(component="attention_value_output", layer=0, moe_key=None),
    )
    assert hook.__self__ is model.model.layers[0].self_attn.o_proj
    assert hook.__name__ == "register_forward_pre_hook"


def test_projection_dimensions_match_actual_gqa_modules(model):
    attention = model.model.layers[0].self_attn
    for component, projection in [
        ("query", attention.q_proj),
        ("key", attention.k_proj),
        ("value", attention.v_proj),
    ]:
        width = projection.weight.shape[0]
        assert (
            get_dimension_by_component(type(model), model.config, component + "_output")
            == width
        )
        heads = (
            model.config.num_attention_heads
            if component == "query"
            else model.config.num_key_value_heads
        )
        head_dim = get_dimension_by_component(
            type(model), model.config, "head_" + component + "_output"
        )
        assert head_dim * heads == width
        rows = torch.arange(2 * 3 * width).reshape(2, 3, width)
        split = output_to_subcomponent(
            rows, "head_" + component + "_output", type(model), model.config
        )
        assert split.shape == (2, heads, 3, head_dim)
        torch.testing.assert_close(split.permute(0, 2, 1, 3).reshape_as(rows), rows)
    assert (
        get_dimension_by_component(type(model), model.config, "attention_value_output")
        == attention.o_proj.weight.shape[1]
    )


def test_unknown_same_named_model_is_not_recognized():
    unknown = type("Qwen2ForCausalLM", (), {"__module__": "unrelated_project"})
    assert get_model_profile(unknown) is None


def test_explicit_component_path_for_custom_model():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    # Module access uses registered attribute names, including numeric ones.
    hook = get_module_hook(
        model, SimpleNamespace(component="0.output", layer=0, moe_key=None)
    )
    assert hook.__self__ is model[0]
    with pytest.raises(ValueError, match="Unknown component"):
        get_module_hook(
            model, SimpleNamespace(component="block_output", layer=0, moe_key=None)
        )


def test_reft_import_does_not_load_unrelated_model_families():
    code = """
import sys
import easysteer.reft.pyreft
families = ('qwen2', 'llama', 'mistral', 'gemma2', 'blip', 'llava', 'esm')
loaded = [name for name in sys.modules if any(
    name == f'transformers.models.{family}.modeling_{family}' for family in families
)]
assert not loaded, loaded
assert 'nnsight' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
