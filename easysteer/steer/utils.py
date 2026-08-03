"""
Utility classes and functions for steering methods
包含StatisticalControlVector类和通用工具函数
"""

import dataclasses
import os
import warnings
from pathlib import Path

import gguf
import numpy as np
import torch

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StatisticalControlVector:
    """Statistical control vector with multi-layer directions"""
    model_type: str
    method: str
    directions: dict[int, np.ndarray]
    metadata: dict = None

    def export_gguf(self, path: os.PathLike[str] | str):
        """
        Export a trained StatisticalControlVector to a llama.cpp .gguf file.
        Compatible with repeng format.
        """
        arch = "controlvector"
        writer = gguf.GGUFWriter(path, arch)
        writer.add_string(f"{arch}.model_hint", self.model_type)
        writer.add_string(f"{arch}.method", self.method)
        writer.add_uint32(f"{arch}.layer_count", len(self.directions))
        
        if self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, (int, float)):
                    writer.add_float32(f"{arch}.{key}", float(value))
                elif isinstance(value, str):
                    writer.add_string(f"{arch}.{key}", value)
                elif isinstance(value, dict):
                    # Handle nested dictionaries like explained_variance
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            writer.add_float32(f"{arch}.{key}.{subkey}", float(subvalue))
        
        for layer in self.directions.keys():
            writer.add_tensor(f"direction.{layer}", self.directions[layer])
        
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "StatisticalControlVector":
        """Import a StatisticalControlVector from a .gguf file"""
        reader = gguf.GGUFReader(path)

        archf = reader.get_field("general.architecture")
        if not archf or not len(archf.parts):
            warnings.warn(".gguf file missing architecture field")
        else:
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "controlvector":
                warnings.warn(
                    f".gguf file with architecture {arch!r} does not appear to be a control vector!"
                )

        modelf = reader.get_field("controlvector.model_hint")
        if not modelf or not len(modelf.parts):
            raise ValueError(".gguf file missing controlvector.model_hint field")
        model_hint = str(bytes(modelf.parts[-1]), encoding="utf-8")

        methodf = reader.get_field("controlvector.method")
        method = "unknown"
        if methodf and len(methodf.parts):
            method = str(bytes(methodf.parts[-1]), encoding="utf-8")

        directions = {}
        metadata = {}
        
        # Extract metadata
        for field_name, field in reader.fields.items():
            if field_name.startswith("controlvector.") and not field_name.endswith((".model_hint", ".method", ".layer_count")):
                key = field_name.replace("controlvector.", "")
                if field.types == [gguf.GGMLQuantizationType.F32]:
                    metadata[key] = float(field.parts[0])
                elif field.types == [gguf.GGMLQuantizationType.I32]:
                    metadata[key] = int(field.parts[0])
        
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f".gguf file has invalid direction field name: {tensor.name}"
                )
            directions[layer] = tensor.data

        return cls(model_type=model_hint, method=method, directions=directions, metadata=metadata)


def extract_token_hiddens(all_hidden_states, positive_indices, negative_indices=None, token_pos=-1):
    """
    从all_hidden_states中提取指定位置token的hidden states
    
    Args:
        all_hidden_states: 三维列表 [样本][layer][token]，其中每个hidden state是tensor或numpy array
        positive_indices: 正样本的索引列表
        negative_indices: 负样本的索引列表，如果为None则自动推断
        token_pos: token位置，支持以下格式：
                  - int: 具体位置索引，-1表示最后一个token（默认）
                  - "first": 第一个token
                  - "last": 最后一个token
                  - "mean": 所有token的均值
                  - "max": 所有token的最大值（按L2范数）
                  - "min": 所有token的最小值（按L2范数）
    
    Returns:
        tuple: (positive_hiddens, negative_hiddens)
               每个都是dict {layer: np.ndarray}，shape为(n_samples, hidden_dim)
    """
    # CaptureResult (easysteer.hidden_states) is accepted directly; it
    # converts to the nested [sample][layer][token] shape with exact,
    # label-driven per-sample rows, and its TRUE layer ids key the
    # output dicts (legacy nested input keeps positional keys).
    layer_keys = None
    if hasattr(all_hidden_states, "to_nested"):
        layer_keys = list(all_hidden_states.layer_ids)
        all_hidden_states = all_hidden_states.to_nested()
    if negative_indices is None:
        # 假设前半部分是positive，后半部分是negative
        n_samples = len(all_hidden_states)
        positive_indices = list(range(n_samples // 2))
        negative_indices = list(range(n_samples // 2, n_samples))
    
    n_layers = len(all_hidden_states[0])  # 层数
    if layer_keys is None:
        layer_keys = list(range(n_layers))

    positive_hiddens = {layer: [] for layer in layer_keys}
    negative_hiddens = {layer: [] for layer in layer_keys}
    
    def extract_token_from_sequence(token_sequence, pos):
        """从token序列中提取指定位置的token"""
        if isinstance(pos, int):
            return token_sequence[pos]
        elif pos == "first":
            return token_sequence[0]
        elif pos == "last":
            return token_sequence[-1]
        elif pos == "mean":
            # 计算所有token的均值
            tokens = np.stack([t.cpu().float().numpy() if torch.is_tensor(t) else t for t in token_sequence])
            return np.mean(tokens, axis=0)
        elif pos == "max":
            # 选择L2范数最大的token
            norms = []
            tokens = []
            for t in token_sequence:
                if torch.is_tensor(t):
                    t = t.cpu().float().numpy()
                tokens.append(t)
                norms.append(np.linalg.norm(t))
            max_idx = np.argmax(norms)
            return tokens[max_idx]
        elif pos == "min":
            # 选择L2范数最小的token
            norms = []
            tokens = []
            for t in token_sequence:
                if torch.is_tensor(t):
                    t = t.cpu().float().numpy()
                tokens.append(t)
                norms.append(np.linalg.norm(t))
            min_idx = np.argmin(norms)
            return tokens[min_idx]
        else:
            raise ValueError(f"Unsupported token_pos: {pos}")
    
    # 提取positive样本的指定token
    for sample_idx in positive_indices:
        sample_hiddens = all_hidden_states[sample_idx]
        for layer_pos in range(n_layers):
            token_hidden = extract_token_from_sequence(sample_hiddens[layer_pos], token_pos)
            # 转换为numpy array
            if torch.is_tensor(token_hidden):
                token_hidden = token_hidden.cpu().float().numpy()
            positive_hiddens[layer_keys[layer_pos]].append(token_hidden)
    
    # 提取negative样本的指定token（只有当negative_indices非空时）
    if negative_indices:
        for sample_idx in negative_indices:
            sample_hiddens = all_hidden_states[sample_idx]
            for layer in range(n_layers):
                token_hidden = extract_token_from_sequence(sample_hiddens[layer], token_pos)
                # 转换为numpy array
                if torch.is_tensor(token_hidden):
                    token_hidden = token_hidden.cpu().float().numpy()
                negative_hiddens[layer].append(token_hidden)
    
    # 转换为numpy arrays
    positive_hiddens = {k: np.vstack(v) for k, v in positive_hiddens.items()}
    # 只在negative_hiddens有数据时才进行vstack
    if negative_indices and any(negative_hiddens.values()):
        negative_hiddens = {k: np.vstack(v) for k, v in negative_hiddens.items()}
    else:
        negative_hiddens = {}
    
    return positive_hiddens, negative_hiddens



