"""Model construction helper for llama."""

import torch


def create_llama(
    name="sharpbai/alpaca-7b-merged", cache_dir=None, dtype=torch.bfloat16, config=None
):
    """Return config, tokenizer and a loaded or freshly initialized Llama model."""
    from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
    if config is None:
        config = LlamaConfig.from_pretrained(name, cache_dir=cache_dir)
        llama = LlamaForCausalLM.from_pretrained(
            name,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        )
    else:
        llama = LlamaForCausalLM(config)
    tokenizer = LlamaTokenizer.from_pretrained(name, cache_dir=cache_dir)
    print("loaded model")
    return config, tokenizer, llama
