# SPDX-License-Identifier: Apache-2.0
"""
Model Structure Detection

Functions for automatically detecting transformer layer structures in models.
"""

from typing import List, Tuple
import torch.nn as nn
from .patterns import get_layer_patterns_for_model


def detect_layer_structure(model: nn.Module) -> List[Tuple[str, str]]:
    """
    自动检测模型的层结构

    Args:
        model: 要检测的模型

    Returns:
        [(层路径, 层类型), ...] 的列表
    """
    layer_info = []

    def analyze_module(module: nn.Module, name: str = "", depth: int = 0):
        # 避免递归过深
        if depth > 10:
            return

        # 检查是否是transformer层
        module_type = type(module).__name__

        # 常见的transformer层类名模式
        transformer_layer_patterns = [
            "TransformerBlock", "DecoderLayer", "EncoderLayer", "Block",
            "LlamaDecoderLayer", "MistralDecoderLayer", "PhiDecoderLayer",
            "ChatGLMBlock", "BaichuanLayer", "QWenBlock", "InternLMDecoderLayer",
            "GPTBlock", "GPTJBlock", "GPTNeoXLayer", "BloomBlock",
            "FalconDecoderLayer", "MPTBlock", "OPTDecoderLayer",
            "T5Block", "BertLayer", "RobertaLayer"
        ]

        # 检查是否匹配transformer层模式
        is_transformer_layer = any(pattern in module_type for pattern in transformer_layer_patterns)

        if is_transformer_layer and hasattr(module, 'forward'):
            layer_info.append((name, module_type))

        # 递归检查子模块
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            analyze_module(child_module, full_name, depth + 1)

    analyze_module(model)
    return layer_info


def get_optimal_layer_pattern(model: nn.Module, model_name: str = "") -> str:
    """
    获取最优的层模式，结合自动检测和模型名称

    Args:
        model: 要分析的模型
        model_name: 模型名称（可选）

    Returns:
        最优的层模式字符串
    """
    # Check if this is a vision-language model that has already been extracted
    # In this case, the model should already be the language_model part
    # VL models will have already been unwrapped by the adapter
    
    # 首先尝试基于模型名称的模式
    if model_name:
        pattern_from_name = get_layer_patterns_for_model(model_name)
    else:
        pattern_from_name = "layers"

    # 自动检测层结构
    detected_layers = detect_layer_structure(model)

    if not detected_layers:
        # 没有检测到层，使用基于名称的模式
        return pattern_from_name

    # 分析检测到的层路径，找出最佳匹配模式
    layer_paths = [path for path, _ in detected_layers]
    
    # 统计不同路径段的出现频率
    path_segments = {}
    for path in layer_paths:
        segments = path.split('.')
        for i in range(len(segments) - 1):  # 排除最后一个段（通常是具体的层索引）
            segment_path = '.'.join(segments[:i + 1])
            path_segments[segment_path] = path_segments.get(segment_path, 0) + 1

    # 找出最频繁出现的路径段
    if path_segments:
        best_pattern = max(path_segments.items(), key=lambda x: x[1])[0]
        # 验证这个模式确实匹配检测到的层
        matching_layers = [path for path in layer_paths if best_pattern in path]
        if len(matching_layers) >= len(detected_layers) * 0.8:  # 至少匹配80%的层
            return best_pattern

    # 如果自动检测失败，回退到基于名称的模式
    return pattern_from_name 