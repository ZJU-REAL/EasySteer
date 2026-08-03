"""
Steering Methods Package
统一的控制向量提取方法包
"""

from .utils import StatisticalControlVector, extract_token_hiddens
from .accumulators import (
    DiffMeanAccumulator,
    MomentsAccumulator,
    TopKCountAccumulator,
)
from .diffmean import DiffMeanExtractor
from .pca import PCAExtractor
from .lat import LATExtractor
from .linear_probe import LinearProbeExtractor
from .sae import SAEFeatureExplorer, search_sae_features, get_sae_feature_explanation, extract_sae_decoder_vector
from .unified_interface import (
    extract_statistical_control_vector,
    extract_diffmean_control_vector,
    extract_pca_control_vector,
    extract_lat_control_vector,
    extract_linear_probe_control_vector
)

__all__ = [
    "MomentsAccumulator",
    "DiffMeanAccumulator",
    "TopKCountAccumulator",
    # Core classes
    'StatisticalControlVector',
    
    # Utility functions
    'extract_token_hiddens',
    
    # Extractor classes
    'DiffMeanExtractor',
    'PCAExtractor',
    'LATExtractor',
    'LinearProbeExtractor',
    'SAEFeatureExplorer',
    
    # Unified interface functions
    'extract_statistical_control_vector',
    'extract_diffmean_control_vector',
    'extract_pca_control_vector',
    'extract_lat_control_vector',
    'extract_linear_probe_control_vector',
    
    # SAE feature exploration functions
    'search_sae_features',
    'get_sae_feature_explanation',
    'extract_sae_decoder_vector'
] 