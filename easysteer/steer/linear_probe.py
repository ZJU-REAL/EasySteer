"""
Linear Probe Extractor
线性探测器方法
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from .utils import StatisticalControlVector

import logging
logger = logging.getLogger(__name__)


class LinearProbeExtractor:
    """Linear Probe method for control vector extraction"""
    
    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        model_type: str = "unknown",
        normalize: bool = True,
        token_pos: int | str = -1,
        regularization: str = "l2",
        C: float = 1.0,
        standardize: bool = True,
        **kwargs
    ) -> StatisticalControlVector:
        """
        Extract control vectors using Linear Probe method.
        
        Args:
            all_hidden_states: 三维列表 [样本][layer][token]
            positive_indices: 正样本的索引列表
            negative_indices: 负样本的索引列表
            model_type: 模型类型名称
            normalize: 是否归一化向量
            token_pos: token位置，-1表示最后一个token（默认），支持int/"first"/"last"/"mean"/"max"/"min"
            regularization: 正则化类型 ("l1", "l2", "elasticnet", "none")
            C: 正则化强度的倒数（值越小正则化越强）
            standardize: 是否标准化特征
        """
        if negative_indices is None:
            n_samples = len(all_hidden_states)
            negative_indices = list(range(len(positive_indices), n_samples))
        
        # 检查样本数量
        total_samples = len(positive_indices) + len(negative_indices)
        if total_samples < 4:
            raise ValueError(
                f"LinearProbe方法至少需要4个样本（2个positive + 2个negative），"
                f"但只提供了{total_samples}个样本。"
            )
        
        if len(positive_indices) < 1 or len(negative_indices) < 1:
            raise ValueError(
                f"LinearProbe方法需要至少1个positive样本和1个negative样本，"
                f"但提供了{len(positive_indices)}个positive样本和{len(negative_indices)}个negative样本。"
            )
        
        # L1正则化的参数建议
        if regularization == "l1" and C <= 1.0:
            logger.warning(
                f"L1正则化使用C={C}可能过强，容易产生全零权重。"
                f"建议尝试更大的C值（如C=10.0或C=100.0）以减少稀疏化程度。"
            )
        
        logger.info(f"LinearProbe: 使用{len(positive_indices)}个positive样本和{len(negative_indices)}个negative样本")
        logger.info(f"正则化方法: {regularization}, C: {C}")
        
        n_layers = len(all_hidden_states[0])
        directions = {}
        model_scores = {}
        
        from .utils import extract_token_hiddens
        
        # 提取指定位置的token hidden states
        positive_hiddens, negative_hiddens = extract_token_hiddens(
            all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
        )
        
        for layer in tqdm(range(n_layers), desc="Computing LinearProbe directions"):
            # 准备训练数据
            X_pos = positive_hiddens[layer]  # [n_positive, hidden_dim]
            X_neg = negative_hiddens[layer]  # [n_negative, hidden_dim]
            
            # 合并数据
            X = np.vstack([X_pos, X_neg])  # [n_total, hidden_dim]
            y = np.hstack([
                np.ones(len(X_pos)),   # positive样本标签为1
                np.zeros(len(X_neg))   # negative样本标签为0
            ])
            
            # 特征标准化
            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            # 设置正则化参数
            penalty_map = {
                "l1": "l1",
                "l2": "l2", 
                "elasticnet": "elasticnet",
                "none": None
            }
            penalty = penalty_map.get(regularization, "l2")
            
            # 训练逻辑回归分类器
            if penalty == "elasticnet":
                clf = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    l1_ratio=0.5,  # elasticnet的l1和l2权重比例
                    solver="saga",
                    max_iter=1000,
                    random_state=42
                )
            elif penalty is None:
                clf = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42
                )
            else:
                solver = "liblinear" if penalty == "l1" else "lbfgs"
                clf = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    solver=solver,
                    max_iter=1000,
                    random_state=42
                )
            
            try:
                clf.fit(X, y)
                
                # 获取分类器权重作为控制向量
                # 权重向量指向positive类别的方向
                direction = clf.coef_[0]  # [hidden_dim]
                
                # 计算分类准确率
                train_score = clf.score(X, y)
                model_scores[layer] = float(train_score)
                
                # 检查权重稀疏性（针对L1正则化）
                non_zero_weights = np.count_nonzero(direction)
                sparsity_ratio = 1.0 - (non_zero_weights / len(direction))
                
                logger.info(f"Layer {layer}: 分类准确率 {train_score:.4f}, 稀疏度 {sparsity_ratio:.3f} ({non_zero_weights}/{len(direction)} 非零权重)")
                
                # 如果所有权重都是零，发出警告
                if non_zero_weights == 0:
                    logger.warning(f"Layer {layer}: 所有权重为零！可能需要调整正则化参数。")
                
                if normalize:
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    else:
                        logger.warning(f"Layer {layer}: 零向量无法归一化")
                
                directions[layer] = direction.astype(np.float32)
                
            except Exception as e:
                logger.warning(f"Layer {layer}: 训练失败，使用零向量。错误: {e}")
                raise RuntimeError(
                    f"linear probe fit failed for layer {layer}: {e}"
                ) from e
                model_scores[layer] = 0.0
        
        metadata = {
            "normalize": normalize,
            "token_pos": token_pos,
            "regularization": regularization,
            "C": C,
            "standardize": standardize,
            "n_positive": len(positive_indices),
            "n_negative": len(negative_indices),
            "classification_scores": model_scores
        }
        
        return StatisticalControlVector(
            model_type=model_type,
            method="linear_probe",
            directions=directions,
            metadata=metadata
        ) 