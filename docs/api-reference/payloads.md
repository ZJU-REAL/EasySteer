# Payload classes (`vllm.steer_vectors`)

Payloads describe weights independently of source file formats. Construct them
from NumPy or PyTorch tensors, or obtain them through an
[`easysteer.vectors` adapter](steer.md#payload-adapters-easysteervectors), then
pass them as `VectorSpec(data=payload, ...)`.

```python
from vllm.steer_vectors import (
    ConceptPair,
    DirectionVector,
    LinearMap,
    LowRankProjector,
    Payload,
    ReftIntervention,
    RouterConfig,
)
```

The tensor payload constructors validate dimensions and finite values and store
contiguous float32 arrays. Engine admission checks component widths and target layers;
worker materialization uses the configured steering dtype. See
[algorithms and payloads](algorithms.md) for transformations and file schemas.

## Common wire representation

All payload classes inherit `Payload.to_wire()`. It returns a dictionary with
`version`, `kind`, named `tensors` (`shape` and raw byte `data`), `extra` metadata,
and a content `sha256`. Use
[`to_json_payload()`](steer.md#easysteer.vectors.to_json_payload) for JSON/HTTP
encoding of those bytes.

::: model_hooks.steering.payloads.Payload
    options:
      heading: vllm.steer_vectors.Payload
      members: [kind, to_wire]
      show_if_no_docstring: true

## Direction weights

`layers` maps true decoder-layer IDs to one-dimensional vectors. Widths may
differ across layers when the model component does. For `attention_add`, each
vector contains concatenated head outputs; obtain their dimensions from capture
layouts. The constructor returns a `DirectionVector`; algorithms determine how
its directions are applied.

::: model_hooks.steering.payloads.DirectionVector
    options:
      heading: vllm.steer_vectors.DirectionVector
      merge_init_into_class: true
      members: [kind, layers]
      show_if_no_docstring: true

## Affine and low-rank maps

For `LinearMap`, `weight` has shape `(hidden, hidden)` and optional `bias` has
shape `(hidden,)`. For `LowRankProjector`, both projectors have shape
`(hidden, rank)`. These payloads do not record layer IDs; specify
`VectorSpec.layers` when applying the returned object.

::: model_hooks.steering.payloads.LinearMap
    options:
      heading: vllm.steer_vectors.LinearMap
      merge_init_into_class: true
      members: [kind, weight, bias]
      show_if_no_docstring: true

::: model_hooks.steering.payloads.LowRankProjector
    options:
      heading: vllm.steer_vectors.LowRankProjector
      merge_init_into_class: true
      members: [kind, projector1, projector2]
      show_if_no_docstring: true

## LoReFT weights

`rotate_layer` has shape `(hidden, rank)`, `learned_source_weight` has shape
`(rank, hidden)`, and optional `learned_source_bias` has shape `(rank,)`.
`layer` records a checkpoint's target. If it is omitted, supply
`VectorSpec.layers`. The returned `ReftIntervention` is accepted by `loreft`.

::: model_hooks.steering.payloads.ReftIntervention
    options:
      heading: vllm.steer_vectors.ReftIntervention
      merge_init_into_class: true
      members: [kind, rotate_layer, learned_source_weight, learned_source_bias, layer]
      show_if_no_docstring: true

## Concept substitution

`h1` and `h2` are `DirectionVector` objects or layer-to-vector dictionaries.
They must have identical layer sets and matching widths at each layer.
The returned `ConceptPair` preserves the two roles explicitly.

::: model_hooks.steering.payloads.ConceptPair
    options:
      heading: vllm.steer_vectors.ConceptPair
      merge_init_into_class: true
      members: [kind, h1, h2]
      show_if_no_docstring: true

## MoE routing configuration

`layers` maps layer IDs to routing dictionaries. The returned `RouterConfig`
normalizes and validates each dictionary:

| Key | Meaning |
|---|---|
| `mode` | `activate` (default), `deactivate`, `soft`, `soft_topk`, or `soft_random`. |
| `expert_ids` | Nonnegative expert IDs affected by the selected mode. |
| `activate_ids`, `deactivate_ids` | Additional explicit lists accepted by hard activation/deactivation modes. |
| `epsilon` | Hard-routing margin, default `0.01`. |
| `lambda` | Soft-routing strength, default `0.5`. |
| `topk` | Positive expert count for `soft_topk`, default `8`. |

```python
router = RouterConfig({10: {"mode": "deactivate", "expert_ids": [1, 3]}})
```

::: model_hooks.steering.payloads.RouterConfig
    options:
      heading: vllm.steer_vectors.RouterConfig
      merge_init_into_class: true
      members: [kind, layers]
      show_if_no_docstring: true

Pass `router` as `VectorSpec.data` with `algorithm="moe_router"`. Supported
graph modes and overrides are listed in the
[algorithm reference](algorithms.md#capability-table).
