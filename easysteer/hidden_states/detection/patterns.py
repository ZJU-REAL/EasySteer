# SPDX-License-Identifier: Apache-2.0
"""
Layer Pattern Detection

Functions for detecting layer patterns in different transformer model architectures.
"""


def get_layer_patterns_for_model(model_name: str) -> str:
    """
    根据模型名称返回对应的层模式

    Args:
        model_name: 模型名称或路径

    Returns:
        对应的层模式字符串
    """
    # 将模型名称转换为小写以便匹配
    model_name_lower = model_name.lower()

    # 定义不同模型的层模式
    patterns = {
        # Llama系列模型
        "llama": "model.layers",
        "llama-2": "model.layers",
        "llama-3": "model.layers",
        "code-llama": "model.layers",
        "vicuna": "model.layers",
        "alpaca": "model.layers",

        # Qwen系列模型
        "qwen": "transformer.h",
        "qwen1.5": "model.layers",
        "qwen2": "model.layers",
        "qwen2.5": "model.layers",
        "qwen2-vl": "model.layers",
        "qwen2.5-vl": "model.layers",
        "qwen-vl": "transformer.h",
        
        # Vision-Language models (language model components)
        "llava": "model.layers",
        "internvl": "model.layers",

        # Mistral系列模型
        "mistral": "model.layers",
        "mixtral": "model.layers",

        # Phi系列模型
        "phi": "model.layers",
        "phi-2": "model.layers",
        "phi-3": "model.layers",

        # ChatGLM系列模型
        "chatglm": "transformer.encoder.layers",
        "chatglm2": "transformer.encoder.layers",
        "chatglm3": "transformer.encoder.layers",

        # Baichuan系列模型
        "baichuan": "model.layers",
        "baichuan2": "model.layers",

        # InternLM系列模型
        "internlm": "model.layers",
        "internlm2": "model.layers",

        # Yi系列模型
        "yi": "model.layers",

        # DeepSeek系列模型
        "deepseek": "model.layers",

        # BERT系列模型
        "bert": "encoder.layer",
        "roberta": "encoder.layer",
        "distilbert": "transformer.layer",

        # GPT系列模型
        "gpt": "transformer.h",
        "gpt2": "transformer.h",
        "gpt-j": "transformer.h",
        "gpt-neox": "gpt_neox.layers",

        # T5系列模型
        "t5": "encoder.block",
        "flan-t5": "encoder.block",

        # Gemma系列模型
        "gemma": "model.layers",

        # Mamba系列模型
        "mamba": "backbone.layers",
        "mamba2": "backbone.layers",

        # OPT系列模型
        "opt": "model.decoder.layers",

        # Falcon系列模型
        "falcon": "transformer.h",

        # BLOOM系列模型
        "bloom": "transformer.h",

        # MPT系列模型
        "mpt": "transformer.blocks",

        # Pythia系列模型
        "pythia": "gpt_neox.layers",

        # StableLM系列模型
        "stablelm": "model.layers",

        # CodeGen系列模型
        "codegen": "transformer.h",

        # SantaCoder系列模型
        "santacoder": "transformer.h",

        # StarCoder系列模型
        "starcoder": "transformer.h",

        # CodeT5系列模型
        "codet5": "encoder.block",
    }

    # 按优先级匹配模式
    for key, pattern in patterns.items():
        if key in model_name_lower:
            return pattern

    # 默认模式，适用于大多数现代transformer模型
    return "layers" 