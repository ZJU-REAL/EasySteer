"""Model construction helper for qwen2."""

import torch


def create_qwen2(
    name="Qwen/Qwen2-7B-beta", cache_dir=None, dtype=torch.bfloat16
):
    """Load a Qwen2 causal LM and return its config, tokenizer and model."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    
    config = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )
    print("loaded model")
    return config, tokenizer, model
