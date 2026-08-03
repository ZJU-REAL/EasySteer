"""
Linear Algebraic Technique (LAT) Extractor
线性代数技术方法
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from .utils import StatisticalControlVector

import logging
logger = logging.getLogger(__name__)


class LATExtractor:
    """Linear Algebraic Technique (LAT) method for control vector extraction"""
    
    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        model_type: str = "unknown",
        n_components: int = 1,
        use_positive_only: bool = True,
        correct_direction: bool = True,
        normalize: bool = True,
        token_pos: int | str = -1,
        **kwargs
    ) -> StatisticalControlVector:
        """
        Extract control vectors using LAT method.
        LAT is PCA over normalized differences of random pairs of activations.
        
        Args:
            all_hidden_states: 三维列表 [样本][layer][token]
            positive_indices: 正样本的索引列表
            negative_indices: 负样本的索引列表
            model_type: 模型类型名称
            n_components: PCA组件数量
            use_positive_only: 是否只使用正样本
            correct_direction: 是否校正向量方向（确保方向从负样本指向正样本）
            normalize: 是否归一化向量
            token_pos: token位置，-1表示最后一个token（默认），支持int/"first"/"last"/"mean"/"max"/"min"
        """
        if use_positive_only:
            sample_indices = positive_indices
        else:
            if negative_indices is None:
                n_samples = len(all_hidden_states)
                negative_indices = list(range(len(positive_indices), n_samples))
            sample_indices = positive_indices + negative_indices
        
        # 检查样本数量是否足够进行LAT计算
        total_samples = len(sample_indices)
        min_pairs_needed = 2  # PCA至少需要2对样本
        max_pairs_possible = total_samples // 2
        
        if total_samples < 4:
            raise ValueError(
                f"LAT方法至少需要4个样本才能产生2对差值进行有效的PCA计算，"
                f"但只提供了{total_samples}个样本。"
                f"请增加样本数量或使用其他方法（如DiffMean）。"
            )
        
        if max_pairs_possible < min_pairs_needed:
            raise ValueError(
                f"LAT方法需要至少{min_pairs_needed}对样本进行PCA计算，"
                f"但{total_samples}个样本最多只能产生{max_pairs_possible}对。"
                f"请增加样本数量或使用其他方法。"
            )
        
        logger.info(f"LAT: 使用{total_samples}个样本，将产生{max_pairs_possible}对差值")
        
        n_layers = len(all_hidden_states[0])
        directions = {}
        explained_variance = {}
        
        from .utils import extract_token_hiddens
        
        # 使用新的token提取函数
        if use_positive_only:
            positive_hiddens, _ = extract_token_hiddens(
                all_hidden_states, sample_indices, [], token_pos=token_pos
            )
        else:
            positive_hiddens, negative_hiddens = extract_token_hiddens(
                all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
            )
            # 合并正负样本
            combined_hiddens = {}
            for layer in positive_hiddens.keys():
                combined_hiddens[layer] = np.vstack([positive_hiddens[layer], negative_hiddens[layer]])
            positive_hiddens = combined_hiddens
        
        for layer in tqdm(list(positive_hiddens.keys()), desc="Computing LAT directions"):
            all_activations = positive_hiddens[layer]
            
            # LAT: 随机配对并计算差值
            logger.info(f"Layer {layer}: Shuffling {all_activations.shape[0]} activations")
            # Permute a copy: never mutate the caller-visible array.
            all_activations = np.random.permutation(all_activations)
            length = all_activations.shape[0] // 2
            differences = all_activations[:length] - all_activations[length:length * 2]
            
            logger.info(f"Layer {layer}: Shuffled and diff'd: {differences.shape[0]} pairs")
            logger.info(f"Layer {layer}: Potential NaNs: {np.isnan(differences).sum()}")
            logger.info(f"Layer {layer}: Potential Infs: {np.isinf(differences).sum()}")
            logger.info(f"Layer {layer}: Range: {differences.min()} to {differences.max()}")
            
            # 归一化差值，避免除零
            norms = np.linalg.norm(differences, axis=1, keepdims=True)
            differences = np.where(norms == 0, 0, differences / norms)
            
            # 对差值执行PCA
            pca = PCA(n_components=min(n_components + 1, differences.shape[0], differences.shape[1]))
            pca.fit(differences)
            
            # 取第一主成分
            first_component = pca.components_[0]
            variance_explained = pca.explained_variance_ratio_[0]
            
            logger.info(f"Layer {layer}: LAT explains {variance_explained:.5%} of the variance")
            
            # 向量方向校正（确保方向从负样本指向正样本）
            if correct_direction and not use_positive_only and negative_indices is not None and len(negative_indices) > 0:
                # 重新提取正负样本数据用于方向校正
                pos_hiddens_orig, neg_hiddens_orig = extract_token_hiddens(
                    all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
                )
                
                pos_activations_layer = pos_hiddens_orig[layer]
                neg_activations_layer = neg_hiddens_orig[layer]

                # 计算范数，用于投影
                vec_norm = np.linalg.norm(first_component)
                if vec_norm > 1e-6:  # 避免除以零
                    # 将正负样本的激活值投影到主成分方向上
                    proj_pos = (pos_activations_layer @ first_component) / vec_norm
                    proj_neg = (neg_activations_layer @ first_component) / vec_norm
                    
                    # 如果正样本的平均投影值小于负样本，说明方向反了，需要翻转
                    if np.mean(proj_pos) < np.mean(proj_neg):
                        first_component *= -1
                        logger.info(f"Layer {layer}: Direction corrected (flipped)")
            
            if normalize:
                norm = np.linalg.norm(first_component)
                if norm > 0:
                    first_component = first_component / norm
            
            directions[layer] = first_component.astype(np.float32)
            explained_variance[layer] = float(variance_explained)
        
        metadata = {
            "normalize": normalize,
            "n_components": n_components,
            "use_positive_only": use_positive_only,
            "correct_direction": correct_direction,
            "token_pos": token_pos,
            "n_samples": len(sample_indices),
            "n_positive": len(positive_indices),
            "n_negative": len(negative_indices) if negative_indices else 0,
            "explained_variance": explained_variance
        }
        
        return StatisticalControlVector(
            model_type=model_type,
            method="lat",
            directions=directions,
            metadata=metadata
        ) 