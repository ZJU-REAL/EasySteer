"""
Principal Component Analysis Extractor
主成分分析方法
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from .utils import StatisticalControlVector

import logging
logger = logging.getLogger(__name__)


class PCAExtractor:
    """Principal Component Analysis method for control vector extraction"""
    
    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        model_type: str = "unknown",
        n_components: int = 1,
        method: str = "standard",  # "standard", "diff", or "center"
        correct_direction: bool = True,
        normalize: bool = True,
        token_pos: int | str = -1,
        **kwargs
    ) -> StatisticalControlVector:
        """
        Extract control vectors using PCA method.
        
        Args:
            all_hidden_states: 三维列表 [样本][layer][token]
            positive_indices: 正样本的索引列表
            negative_indices: 负样本的索引列表
            model_type: 模型类型名称
            n_components: PCA组件数量
            method: PCA方法类型，可选值为：
                    "standard" - 标准PCA（仅使用正样本）
                    "diff" - 差分PCA（需要正负样本）
                    "center" - 中心化PCA（需要正负样本）
            correct_direction: 是否校正向量方向（确保方向从负样本指向正样本）
            normalize: 是否归一化向量
            token_pos: token位置，-1表示最后一个token（默认），支持int/"first"/"last"/"mean"/"max"/"min"
        """
        # 对于diff和center方法，确保负样本索引存在
        if method in ["diff", "center"]:
            if negative_indices is None:
                n_samples = len(all_hidden_states)
                negative_indices = list(range(len(positive_indices), n_samples))
                if len(negative_indices) == 0:
                    raise ValueError(f"{method}方法需要负样本，但未提供负样本索引")

        n_layers = len(all_hidden_states[0])
        directions = {}
        explained_variance = {}
        
        from .utils import extract_token_hiddens
        
        # 提取指定位置的token hidden states
        positive_hiddens, negative_hiddens = extract_token_hiddens(
            all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
        )
        
        for layer in tqdm(range(n_layers), desc="Computing PCA directions"):
            if method == "standard":
                # 标准PCA（只使用正样本）
                all_activations = positive_hiddens[layer]
                
                if not isinstance(all_activations, np.ndarray):
                    all_activations = np.vstack(all_activations)
                
                # 执行PCA
                pca = PCA(n_components=1)
                pca.fit(all_activations)
                
                # 取第一主成分
                first_component = pca.components_[0]
                variance_explained = pca.explained_variance_ratio_[0]
                
                logger.info(f"Layer {layer}: PCA explains {variance_explained:.5%} of the variance")
                
            elif method == "center":
                # 中心化PCA（需要正负样本）
                pos_activations = positive_hiddens[layer]  # [n_pos, hidden_dim]
                neg_activations = negative_hiddens[layer]  # [n_neg, hidden_dim]
                
                # 确保正负样本数量相等
                min_samples = min(len(pos_activations), len(neg_activations))
                pos_activations = pos_activations[:min_samples]
                neg_activations = neg_activations[:min_samples]
                
                # 计算中心点（正负样本的平均值）
                centers = (pos_activations + neg_activations) / 2
                
                # 中心化数据
                centered_pos = pos_activations - centers
                centered_neg = neg_activations - centers
                
                # 合并所有中心化数据
                all_activations = np.vstack([centered_pos, centered_neg])
                
                # 执行PCA
                pca = PCA(n_components=1)
                pca.fit(all_activations)
                
                # 取第一主成分
                first_component = pca.components_[0]
                variance_explained = pca.explained_variance_ratio_[0]
                
                logger.info(f"Layer {layer}: PCA on centered data explains {variance_explained:.5%} of the variance")
                
            elif method == "diff":
                # 差值PCA（需要正负样本）
                pos_activations = positive_hiddens[layer]  # [n_pos, hidden_dim]
                neg_activations = negative_hiddens[layer]  # [n_neg, hidden_dim]
                
                # 计算每对正负样本的差值
                min_samples = min(len(pos_activations), len(neg_activations))
                differences = []
                
                for i in range(min_samples):
                    diff = pos_activations[i] - neg_activations[i]
                    differences.append(diff)
                
                # 如果正样本更多，添加剩余正样本与负样本均值的差
                if len(pos_activations) > min_samples:
                    neg_mean = np.mean(neg_activations, axis=0)
                    for i in range(min_samples, len(pos_activations)):
                        diff = pos_activations[i] - neg_mean
                        differences.append(diff)
                
                # 如果负样本更多，添加正样本均值与剩余负样本的差
                if len(neg_activations) > min_samples:
                    pos_mean = np.mean(pos_activations, axis=0)
                    for i in range(min_samples, len(neg_activations)):
                        diff = pos_mean - neg_activations[i]
                        differences.append(diff)
                
                all_activations = np.vstack(differences)
                
                # 对差值执行PCA
                pca = PCA(n_components=1)
                pca.fit(all_activations)
                
                first_component = pca.components_[0]
                variance_explained = pca.explained_variance_ratio_[0]
                
                logger.info(f"Layer {layer}: PCA on differences explains {variance_explained:.5%} of the variance")

            # 向量方向校正（确保方向从负样本指向正样本）
            if correct_direction and method in ["diff", "center"]:
                pos_activations_layer = positive_hiddens[layer]
                neg_activations_layer = negative_hiddens[layer]

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
            "method": method,
            "correct_direction": correct_direction,
            "token_pos": token_pos,
            "n_positive": len(positive_indices),
            "n_negative": len(negative_indices) if negative_indices else 0,
            "explained_variance": explained_variance
        }
        
        return StatisticalControlVector(
            model_type=model_type,
            method=f"pca_{method}",
            directions=directions,
            metadata=metadata
        ) 

    @staticmethod
    def from_moments(
        moments,
        model_type: str = "unknown",
        normalize: bool = True,
        pos_moments=None,
        neg_moments=None,
    ) -> StatisticalControlVector:
        """Standard PCA from a streaming second-moment accumulator.

        moments must be a MomentsAccumulator(track_second_moment=True)
        fed the positive rows; the direction per layer is the top
        eigenvector of the covariance. When pos_moments/neg_moments
        (first-moment accumulators) are given, the sign is corrected so
        the direction points from the negative mean to the positive
        mean, matching extract(method="standard", correct_direction).
        """
        directions = {}
        variance = {}
        if not moments.layers:
            raise ValueError("accumulator holds no layers")
        for layer in moments.layers:
            cov = moments.covariance(layer)
            eigvals, eigvecs = np.linalg.eigh(cov)
            direction = eigvecs[:, -1]
            total = float(eigvals.sum())
            variance[layer] = float(eigvals[-1] / total) if total > 0 else 0.0
            if pos_moments is not None and neg_moments is not None:
                gap = pos_moments.mean(layer) - neg_moments.mean(layer)
                if float(gap @ direction) < 0:
                    direction = -direction
            if normalize:
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
            directions[layer] = direction.astype(np.float32)
        return StatisticalControlVector(
            model_type=model_type,
            method="pca_standard",
            directions=directions,
            metadata={
                "normalized": normalize,
                "explained_variance": variance,
                "streaming": True,
            },
        )
