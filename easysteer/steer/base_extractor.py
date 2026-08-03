"""Shared scaffolding for the statistical control-vector extractors.

Every extractor follows the same recipe: resolve the sample split, pull
one token row per sample and layer, compute a per-layer direction,
normalize it, and package everything into a StatisticalControlVector.
`BaseExtractor` owns that recipe; each concrete extractor implements
only the per-layer `_direction()` hook, and its public `extract()`
wrapper validates method-specific options before delegating.
"""

import abc
from typing import ClassVar

import numpy as np
from tqdm.auto import tqdm

from .utils import (
    StatisticalControlVector,
    _metadata,
    extract_token_hiddens,
    l2_normalize,
)


class BaseExtractor(abc.ABC):
    """Template for per-layer statistical control-vector extraction.

    Subclasses set `method` and `progress_desc` and implement
    `_direction()`; their public `extract()` wrappers keep their exact
    historical signatures and call `_extract_template()`.
    """

    method: ClassVar[str]
    """Method name recorded on extracted control vectors."""

    progress_desc: ClassVar[str]
    """Progress-bar label for the per-layer loop."""

    @staticmethod
    @abc.abstractmethod
    def _direction(pos_rows, neg_rows, *, layer, **opts):
        """Compute one layer's raw (un-normalized) direction.

        Args:
            pos_rows (np.ndarray): Positive activations, `(n_pos, dim)`.
            neg_rows (np.ndarray | None): Negative activations,
                `(n_neg, dim)`, or None when no negatives were
                extracted for this run.
            layer (int): Layer key (true layer id for CaptureResult
                input, positional index otherwise).
            **opts (Any): Method-specific options forwarded verbatim
                from `_extract_template()`.

        Returns:
            tuple[np.ndarray, dict]: The direction and per-layer
                metadata extras (e.g. explained variance) keyed by
                metadata field name; `{}` when there are none.
        """

    @classmethod
    def _extract_template(
        cls,
        all_hidden_states,
        positive_indices,
        negative_indices,
        *,
        model_type,
        normalize,
        token_pos,
        extraction_negatives=None,
        opts=None,
        extra_metadata=None,
        method=None,
    ) -> StatisticalControlVector:
        """Run the shared extraction recipe.

        The corpus is tokenized into per-layer row matrices exactly
        once; `_direction()` is then invoked per layer, its result
        optionally L2-normalized, cast to float32, and packaged with
        canonical metadata.

        Args:
            all_hidden_states (list | CaptureResult): Nested
                `[sample][layer][token]` hidden states, or a
                CaptureResult from easysteer.hidden_states.
            positive_indices (list[int]): Indices of positive samples.
            negative_indices (list[int] | None): Resolved negative
                indices; used for the metadata count and, by default,
                for extraction.
            model_type (str): Model type name recorded in the result.
            normalize (bool): Normalize each direction to unit L2 norm.
            token_pos (int | str): Token position passed to
                `extract_token_hiddens`.
            extraction_negatives (list[int] | None): Negative indices
                to actually extract. None (default) extracts
                ``negative_indices``; pass `[]` when the method ignores
                negative rows.
            opts (dict | None): Keyword options forwarded to
                `_direction()`.
            extra_metadata (dict | None): Method-specific metadata
                entries recorded verbatim.
            method (str | None): Override for the recorded method name;
                defaults to `cls.method`.

        Returns:
            StatisticalControlVector: The extracted control vector.
        """
        opts = opts or {}
        if extraction_negatives is None:
            extraction_negatives = negative_indices or []

        positive_hiddens, negative_hiddens = extract_token_hiddens(
            all_hidden_states,
            positive_indices,
            extraction_negatives,
            token_pos=token_pos,
        )

        directions = {}
        layer_stats: dict[str, dict] = {}
        for layer in tqdm(list(positive_hiddens), desc=cls.progress_desc):
            neg_rows = negative_hiddens[layer] if negative_hiddens else None
            direction, extras = cls._direction(
                positive_hiddens[layer], neg_rows, layer=layer, **opts
            )
            if normalize:
                direction = l2_normalize(direction)
            directions[layer] = direction.astype(np.float32)
            for key, value in extras.items():
                layer_stats.setdefault(key, {})[layer] = value

        metadata = _metadata(
            normalize=normalize,
            n_positive=len(positive_indices),
            n_negative=len(negative_indices) if negative_indices else 0,
            token_pos=token_pos,
            **(extra_metadata or {}),
            **layer_stats,
        )
        return StatisticalControlVector(
            model_type=model_type,
            method=method or cls.method,
            directions=directions,
            metadata=metadata,
        )
