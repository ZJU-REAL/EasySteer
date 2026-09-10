"""Shared per-layer extraction and control-vector assembly."""

import abc
from typing import ClassVar

import numpy as np
from tqdm.auto import tqdm

from .utils import (
    StatisticalControlVector,
    _metadata,
    iter_token_hiddens,
    l2_normalize,
)


class BaseExtractor(abc.ABC):
    """Template for per-layer statistical control-vector extraction.

    Subclasses set `method` and `progress_desc` and implement
    `_direction()`. Their public `extract()` methods validate options
    before calling `_extract_template()`.
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
        normalize,
        token_pos,
        extraction_negatives=None,
        opts=None,
        extra_metadata=None,
        method=None,
    ) -> StatisticalControlVector:
        """Select token rows, compute directions, and assemble a control vector.

        extraction_negatives overrides the negative rows used for computation;
        negative_indices still determines the sample count in metadata.
        """
        opts = opts or {}
        if extraction_negatives is None:
            extraction_negatives = negative_indices or []

        layer_rows = iter_token_hiddens(
            all_hidden_states,
            positive_indices,
            extraction_negatives,
            token_pos=token_pos,
        )

        directions = {}
        layer_stats: dict[str, dict] = {}
        for layer, pos_rows, neg_rows in tqdm(layer_rows, desc=cls.progress_desc):
            direction, extras = cls._direction(pos_rows, neg_rows, layer=layer, **opts)
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
            method=method or cls.method,
            directions=directions,
            metadata=metadata,
        )
