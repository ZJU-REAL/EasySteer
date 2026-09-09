"""Control-vector extraction methods, loaded when requested."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .accumulators import (
        DiffMeanAccumulator as DiffMeanAccumulator,
    )
    from .accumulators import (
        MomentsAccumulator as MomentsAccumulator,
    )
    from .accumulators import (
        TopKCountAccumulator as TopKCountAccumulator,
    )
    from .diffmean import DiffMeanExtractor as DiffMeanExtractor
    from .lat import LATExtractor as LATExtractor
    from .linear_probe import LinearProbeExtractor as LinearProbeExtractor
    from .pca import PCAExtractor as PCAExtractor
    from .sae import (
        SAEFeatureExplorer as SAEFeatureExplorer,
    )
    from .sae import (
        extract_sae_decoder_vector as extract_sae_decoder_vector,
    )
    from .sae import (
        get_sae_feature_explanation as get_sae_feature_explanation,
    )
    from .sae import (
        search_sae_features as search_sae_features,
    )
    from .unified_interface import (
        extract_diffmean_control_vector as extract_diffmean_control_vector,
    )
    from .unified_interface import (
        extract_lat_control_vector as extract_lat_control_vector,
    )
    from .unified_interface import (
        extract_linear_probe_control_vector as extract_linear_probe_control_vector,
    )
    from .unified_interface import (
        extract_pca_control_vector as extract_pca_control_vector,
    )
    from .unified_interface import (
        extract_statistical_control_vector as extract_statistical_control_vector,
    )
    from .utils import (
        StatisticalControlVector as StatisticalControlVector,
    )
    from .utils import (
        derive_negative_indices as derive_negative_indices,
    )
    from .utils import (
        extract_token_hiddens as extract_token_hiddens,
    )

_EXPORTS = {
    "MomentsAccumulator": "accumulators",
    "DiffMeanAccumulator": "accumulators",
    "TopKCountAccumulator": "accumulators",
    "StatisticalControlVector": "utils",
    "extract_token_hiddens": "utils",
    "derive_negative_indices": "utils",
    "DiffMeanExtractor": "diffmean",
    "PCAExtractor": "pca",
    "LATExtractor": "lat",
    "LinearProbeExtractor": "linear_probe",
    "SAEFeatureExplorer": "sae",
    "extract_statistical_control_vector": "unified_interface",
    "extract_diffmean_control_vector": "unified_interface",
    "extract_pca_control_vector": "unified_interface",
    "extract_lat_control_vector": "unified_interface",
    "extract_linear_probe_control_vector": "unified_interface",
    "search_sae_features": "sae",
    "get_sae_feature_explanation": "sae",
    "extract_sae_decoder_vector": "sae",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
