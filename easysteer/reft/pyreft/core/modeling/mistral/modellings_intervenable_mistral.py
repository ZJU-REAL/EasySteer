"""Model construction helper for mistral."""

import torch


def create_mistral(
    name="mistralai/Mistral-7B-v0.1", cache_dir=None
):
    """Load a Mistral causal LM and return its config, tokenizer and model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    mistral = AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    )
    print("loaded model")
    return config, tokenizer, mistral
