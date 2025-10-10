# SPDX-License-Identifier: Apache-2.0
"""
Transformer Layer Wrappers

This module implements wrapper layers that can capture hidden states from transformer layers
without modifying the original model implementations.
"""

from typing import Optional
import torch
from torch import nn
from .storage import get_global_store


class TransformerLayerWrapper(nn.Module):
    """
    包装transformer层以捕获hidden states

    这个包装器会捕获每一层的最终输出，即经过了attention、MLP和残差连接后的完整累积状态
    """

    def __init__(self, base_layer: nn.Module, layer_id: int, layer_name: str = "", 
                 store=None) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.layer_id = layer_id
        self.layer_name = layer_name or f"layer_{layer_id}"
        self.store = store if store is not None else get_global_store()

    def forward(self, *args, **kwargs):
        """
        Forward pass that captures the final hidden state output
        """
        # 调用原始层的forward方法
        output = self.base_layer(*args, **kwargs)

        # 处理不同的返回格式并提取hidden states
        hidden_states = self._extract_hidden_states(output)

        # 存储最终的hidden states（这是每一层经过attention、MLP和残差连接后的完整累积状态）
        if hidden_states is not None:
            self.store.store_hidden_state(
                self.layer_id,
                hidden_states,
                self.layer_name
            )

        return output

    def _extract_hidden_states(self, output) -> Optional[torch.Tensor]:
        """
        从层输出中提取hidden states

        Args:
            output: 层的输出，可能是tensor、tuple或其他格式

        Returns:
            提取的hidden states tensor，如果无法提取则返回None
        """
        if isinstance(output, torch.Tensor):
            # 直接返回tensor的情况
            return output
        elif isinstance(output, tuple):
            # 返回tuple的情况，需要检查是否是(hidden_states, residual)格式
            if len(output) >= 2 and isinstance(output[0], torch.Tensor) and isinstance(output[1], torch.Tensor):
                hidden_states, residual = output[0], output[1]

                # 检查两个tensor的形状是否兼容（可以相加）
                if hidden_states.shape == residual.shape:
                    # 这很可能是(hidden_states, residual)的情况，需要相加得到完整的hidden states
                    return hidden_states + residual
                else:
                    # 形状不匹配，可能是其他类型的tuple，使用第一个元素
                    return hidden_states
            elif len(output) > 0 and isinstance(output[0], torch.Tensor):
                # 只有第一个元素是tensor的情况
                return output[0]
        elif isinstance(output, dict):
            # 返回字典的情况，查找常见的键名
            for key in ['hidden_states', 'last_hidden_state', 'output']:
                if key in output and isinstance(output[key], torch.Tensor):
                    return output[key]
        elif hasattr(output, 'hidden_states'):
            # 如果有hidden_states属性
            if isinstance(output.hidden_states, torch.Tensor):
                return output.hidden_states
        elif hasattr(output, 'last_hidden_state'):
            # 如果有last_hidden_state属性
            if isinstance(output.last_hidden_state, torch.Tensor):
                return output.last_hidden_state

        # 无法提取时返回None
        return None

    def __getattr__(self, name):
        """Delegate attribute access to the base layer"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)


def wrap_transformer_layers(model: nn.Module, layer_pattern: str = "layers", 
                           store=None) -> nn.Module:
    """
    自动包装模型中的transformer层以捕获hidden states

    Args:
        model: 要包装的模型
        layer_pattern: 层的模式名称，用于识别transformer层
        store: HiddenStatesStore instance (uses global store if None)

    Returns:
        包装后的模型
    """
    # Use global store if not provided
    if store is None:
        store = get_global_store()
    
    layer_id = 0
    wrapped_count = 0

    def wrap_module(module: nn.Module, name: str = "") -> nn.Module:
        nonlocal layer_id, wrapped_count

        # 检查是否是transformer层
        # 更精确的匹配：确保是实际的层而不是容器
        if (layer_pattern in name and
                hasattr(module, 'forward') and
                not isinstance(module, (nn.ModuleList, nn.Sequential)) and
                '.' in name and  # 确保不是顶层模块
                name.count('.') >= 2):  # 确保有足够的层级深度

            # 检查是否已经被包装
            if not isinstance(module, TransformerLayerWrapper):
                wrapped_layer = TransformerLayerWrapper(
                    module, layer_id, name, store
                )
                layer_id += 1
                wrapped_count += 1
                return wrapped_layer

        # 递归处理子模块
        for child_name, child_module in list(module.named_children()):
            full_child_name = f"{name}.{child_name}" if name else child_name
            wrapped_child = wrap_module(child_module, full_child_name)
            if wrapped_child is not child_module:
                setattr(module, child_name, wrapped_child)

        return module

    wrapped_model = wrap_module(model)
    return wrapped_model 