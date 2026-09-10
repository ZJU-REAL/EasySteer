# SPDX-License-Identifier: Apache-2.0
"""ITI directions for attention head outputs before the output projection."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .utils import (
    StatisticalControlVector,
    _metadata,
    derive_negative_indices,
    extract_token_hiddens,
    l2_normalize,
)


def _labelled_rows(captured, positives, negatives, token_pos):
    if negatives is None:
        negatives = derive_negative_indices(len(captured), positives)
    indices = list(positives) + list(negatives)
    if (
        not positives
        or not negatives
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= len(captured) for i in indices)
    ):
        raise ValueError(
            "positive and negative indices must be nonempty, disjoint and valid"
        )
    pos, neg = extract_token_hiddens(captured, positives, negatives, token_pos)
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    return {layer: np.vstack([pos[layer], neg[layer]]) for layer in pos}, labels


class ITIExtractor:
    """Select heads with probes and build mean-difference ITI directions.

    Probes fit training rows and rank heads by validation accuracy. Following
    ITI, the final direction and its projection standard deviation use both
    development splits. Split by question before capture; no held-out test
    answers should appear in either input.
    """

    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        *,
        validation_hidden_states,
        validation_positive_indices,
        validation_negative_indices=None,
        num_heads: dict[int, int],
        top_k: int = 1,
        token_pos: int | str = -1,
    ) -> StatisticalControlVector:
        """Return concatenated per-layer ``sigma * direction`` vectors.

        Inputs use the other extractors' CaptureResult or nested-list layout.
        ``num_heads`` maps layer ids to query head counts; for capture input,
        use ``{layer: layout['num_heads'] for layer, layout in result.layouts.items()}``.
        Each row must contain contiguous head outputs. Unselected head slices
        remain zero; layers with no selected head are omitted. Apply with
        ``attention_add`` and no normalization, using ``scale`` for ITI's alpha.
        """
        train, train_labels = _labelled_rows(
            all_hidden_states, positive_indices, negative_indices, token_pos
        )
        validation, validation_labels = _labelled_rows(
            validation_hidden_states,
            validation_positive_indices,
            validation_negative_indices,
            token_pos,
        )
        if not train or set(train) != set(validation) or set(train) != set(num_heads):
            raise ValueError(
                "training, validation and num_heads must have the same layers"
            )
        if any(type(count) is not int or count < 1 for count in num_heads.values()):
            raise ValueError("num_heads must contain positive query head counts")
        if type(top_k) is not int or not 1 <= top_k <= sum(num_heads.values()):
            raise ValueError("top_k must be between 1 and the total query head count")

        candidates = []
        for layer in sorted(train):
            width = train[layer].shape[1]
            count = num_heads[layer]
            if validation[layer].shape[1] != width or width == 0 or width % count:
                raise ValueError(f"layer {layer}: incompatible attention head widths")
            head_size = width // count
            for head in range(count):
                head_slice = slice(head * head_size, (head + 1) * head_size)
                probe = LogisticRegression(max_iter=1000, random_state=42).fit(
                    train[layer][:, head_slice], train_labels
                )
                score = float(
                    probe.score(validation[layer][:, head_slice], validation_labels)
                )
                candidates.append((score, layer, head, head_slice))

        directions, scores, scales = {}, {}, {}
        labels = np.concatenate([train_labels, validation_labels])
        # A fixed layer/head ordering resolves validation-score ties.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        for score, layer, head, head_slice in candidates[:top_k]:
            rows = np.vstack(
                [train[layer][:, head_slice], validation[layer][:, head_slice]]
            )
            direction = l2_normalize(
                rows[labels == 1].mean(0) - rows[labels == 0].mean(0)
            )
            sigma = float(np.std(rows @ direction))
            if not np.isfinite(sigma) or sigma == 0:
                raise ValueError(
                    f"layer {layer}, head {head}: no nonzero finite ITI direction"
                )
            vector = directions.setdefault(
                layer, np.zeros(train[layer].shape[1], dtype=np.float32)
            )
            vector[head_slice] = sigma * direction
            key = f"{layer}.{head}"
            scores[key], scales[key] = score, sigma

        return StatisticalControlVector(
            method="iti",
            directions=directions,
            metadata=_metadata(
                normalize=False,
                n_positive=int(np.sum(train_labels)),
                n_negative=int(np.sum(1 - train_labels)),
                n_validation_positive=int(np.sum(validation_labels)),
                n_validation_negative=int(np.sum(1 - validation_labels)),
                token_pos=token_pos,
                top_k=top_k,
                num_heads=num_heads,
                selected_head_scores=scores,
                projection_std=scales,
                direction_data="train+validation",
            ),
        )
