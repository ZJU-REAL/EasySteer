# SPDX-License-Identifier: Apache-2.0
"""ITI directions for attention head outputs before the output projection."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from ._utils import _metadata, l2_normalize
from .result import StatisticalControlVector
from .selection import iter_token_hiddens, validate_sample_groups


def _layer_ids(captured):
    if hasattr(captured, "layer_ids"):
        return list(captured.layer_ids)
    return list(range(len(captured[0]))) if len(captured) else []


def _head_counts(training, validation, layers, num_heads):
    inferred = []
    for captured in (training, validation):
        component = getattr(captured, "component", None)
        if component is not None and component != "attention_heads":
            raise ValueError("ITI requires attention_heads captures")
        layouts = getattr(captured, "layouts", {})
        if layouts:
            if set(layouts) != set(layers) or any(
                "num_heads" not in layout for layout in layouts.values()
            ):
                raise ValueError("ITI requires query head counts for every layer")
            inferred.append({layer: layouts[layer]["num_heads"] for layer in layers})
    if num_heads is None:
        if not inferred:
            raise ValueError(
                "num_heads is required when capture layouts are unavailable"
            )
        num_heads = inferred[0]
    if set(num_heads) != set(layers):
        raise ValueError("training, validation and num_heads must have the same layers")
    if any(type(count) is not int or count < 1 for count in num_heads.values()):
        raise ValueError("num_heads must contain positive query head counts")
    if any(counts != num_heads for counts in inferred):
        raise ValueError(
            "num_heads disagrees with training or validation capture layouts"
        )
    return dict(num_heads)


def _layer_rows(captured, positives, negatives, token_pos, layer):
    _, positive, negative = next(
        iter_token_hiddens(captured, positives, negatives, token_pos, layers=[layer])
    )
    return np.vstack([positive, negative])


def _check_layout(captured, layer, width, count):
    layout = getattr(captured, "layouts", {}).get(layer)
    if layout is not None and (
        layout.get("width") != width or layout.get("head_size") != width // count
    ):
        raise ValueError(f"layer {layer}: incompatible attention head widths")


class ITIExtractor:
    """Select heads with probes and build mean-difference ITI directions.

    Probes fit training rows and rank heads by validation accuracy. Following
    ITI, the final direction and its projection standard deviation use both
    development splits. Split by question before capture; no held-out test
    answers should appear in either input. Only one layer of training and
    validation features is materialized at a time. A second pass reads the
    layers containing selected heads.
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
        num_heads: dict[int, int] | None = None,
        top_k: int = 1,
        token_pos: int | str = -1,
    ) -> StatisticalControlVector:
        """Return concatenated per-layer ``sigma * direction`` vectors.

        Inputs use the other extractors' CaptureResult or nested-list layout.
        ``num_heads`` maps layer ids to query head counts and is inferred from
        capture layouts when omitted. Explicit counts must agree with capture
        layouts. Each row must contain contiguous head outputs. Unselected
        head slices remain zero; layers with no selected head are omitted.
        Apply with ``attention_add`` and no normalization, using ``scale`` for
        ITI's alpha.
        """
        positive_indices, negative_indices = validate_sample_groups(
            len(all_hidden_states), positive_indices, negative_indices
        )
        validation_positive_indices, validation_negative_indices = (
            validate_sample_groups(
                len(validation_hidden_states),
                validation_positive_indices,
                validation_negative_indices,
            )
        )
        layers = _layer_ids(all_hidden_states)
        if not layers or set(layers) != set(_layer_ids(validation_hidden_states)):
            raise ValueError("training and validation must have the same layers")
        num_heads = _head_counts(
            all_hidden_states, validation_hidden_states, layers, num_heads
        )
        if type(top_k) is not int or not 1 <= top_k <= sum(num_heads.values()):
            raise ValueError("top_k must be between 1 and the total query head count")
        train_labels = np.concatenate(
            [np.ones(len(positive_indices)), np.zeros(len(negative_indices))]
        )
        validation_labels = np.concatenate(
            [
                np.ones(len(validation_positive_indices)),
                np.zeros(len(validation_negative_indices)),
            ]
        )

        def read_layer(layer):
            train = _layer_rows(
                all_hidden_states, positive_indices, negative_indices, token_pos, layer
            )
            validation = _layer_rows(
                validation_hidden_states,
                validation_positive_indices,
                validation_negative_indices,
                token_pos,
                layer,
            )
            return train, validation

        candidates = []
        for layer in sorted(layers):
            train, validation = read_layer(layer)
            width = train.shape[1]
            count = num_heads[layer]
            if validation.shape[1] != width or width % count:
                raise ValueError(f"layer {layer}: incompatible attention head widths")
            _check_layout(all_hidden_states, layer, width, count)
            _check_layout(validation_hidden_states, layer, width, count)
            head_size = width // count
            for head in range(count):
                head_slice = slice(head * head_size, (head + 1) * head_size)
                probe = LogisticRegression(max_iter=1000, random_state=42).fit(
                    train[:, head_slice], train_labels
                )
                score = float(probe.score(validation[:, head_slice], validation_labels))
                candidates.append((score, layer, head, head_slice))
            del train, validation, probe

        # A fixed layer/head ordering resolves validation-score ties.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = {}
        for score, layer, head, head_slice in candidates[:top_k]:
            selected.setdefault(layer, []).append((score, head, head_slice))

        directions, scores, scales = {}, {}, {}
        labels = np.concatenate([train_labels, validation_labels])
        for layer in sorted(selected):
            train, validation = read_layer(layer)
            vector = np.zeros(train.shape[1], dtype=np.float32)
            for score, head, head_slice in selected[layer]:
                rows = np.vstack([train[:, head_slice], validation[:, head_slice]])
                direction = l2_normalize(
                    rows[labels == 1].mean(0) - rows[labels == 0].mean(0)
                )
                sigma = float(np.std(rows @ direction))
                if not np.isfinite(sigma) or sigma == 0:
                    raise ValueError(
                        f"layer {layer}, head {head}: no nonzero finite ITI direction"
                    )
                vector[head_slice] = sigma * direction
                key = f"{layer}.{head}"
                scores[key], scales[key] = score, sigma
                del rows
            directions[layer] = vector
            del train, validation

        return StatisticalControlVector(
            method="iti",
            directions=directions,
            component=getattr(all_hidden_states, "component", None),
            model_type=getattr(all_hidden_states, "model", None) or "unknown",
            metadata=_metadata(
                normalize=False,
                n_positive=len(positive_indices),
                n_negative=len(negative_indices),
                n_validation_positive=len(validation_positive_indices),
                n_validation_negative=len(validation_negative_indices),
                token_pos=token_pos,
                top_k=top_k,
                num_heads=num_heads,
                selected_head_scores=scores,
                projection_std=scales,
                direction_data="train+validation",
            ),
        )
