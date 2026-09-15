# SPDX-License-Identifier: Apache-2.0
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
    from .api import (
        extract_diffmean_control_vector as extract_diffmean_control_vector,
    )
    from .api import (
        extract_lat_control_vector as extract_lat_control_vector,
    )
    from .api import (
        extract_linear_probe_control_vector as extract_linear_probe_control_vector,
    )
    from .api import (
        extract_pca_control_vector as extract_pca_control_vector,
    )
    from .api import (
        extract_statistical_control_vector as extract_statistical_control_vector,
    )
    from .diffmean import DiffMeanExtractor as DiffMeanExtractor
    from .iti import ITIExtractor as ITIExtractor
    from .lat import LATExtractor as LATExtractor
    from .linear_probe import LinearProbeExtractor as LinearProbeExtractor
    from .pca import PCAExtractor as PCAExtractor
    from .result import (
        StatisticalControlVector as StatisticalControlVector,
    )
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
    from .selection import (
        derive_negative_indices as derive_negative_indices,
    )
    from .selection import (
        extract_token_hiddens as extract_token_hiddens,
    )

_EXPORTS = {
    "MomentsAccumulator": "accumulators",
    "DiffMeanAccumulator": "accumulators",
    "TopKCountAccumulator": "accumulators",
    "StatisticalControlVector": "result",
    "extract_token_hiddens": "selection",
    "derive_negative_indices": "selection",
    "DiffMeanExtractor": "diffmean",
    "ITIExtractor": "iti",
    "PCAExtractor": "pca",
    "LATExtractor": "lat",
    "LinearProbeExtractor": "linear_probe",
    "SAEFeatureExplorer": "sae",
    "extract_statistical_control_vector": "api",
    "extract_diffmean_control_vector": "api",
    "extract_pca_control_vector": "api",
    "extract_lat_control_vector": "api",
    "extract_linear_probe_control_vector": "api",
    "search_sae_features": "sae",
    "get_sae_feature_explanation": "sae",
    "extract_sae_decoder_vector": "sae",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
