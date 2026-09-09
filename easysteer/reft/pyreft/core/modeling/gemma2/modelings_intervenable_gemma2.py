"""Model construction helper for gemma2."""

import torch


def create_gemma2(
    name="google/gemma2-2b", cache_dir=None, dtype=torch.bfloat16
):
    """Load a Gemma2 causal LM and return its config, tokenizer and model."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    gemma = AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )
    print("loaded model")
    return config, tokenizer, gemma
