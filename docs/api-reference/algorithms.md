# Algorithms and payloads

An algorithm determines both the transformation and the model component it edits.
The payload stores weights; `VectorSpec` supplies the scale, target layers, and
token selection. Declare the algorithm at engine startup, then attach the spec
to each request as shown in the [steering guide](../user-guide/steering.md).

## Capability table

Every registered algorithm supports eager and `split` execution on a model
that exposes the required component. `in_graph` accepts one vector per request
and applies the additional payload conditions below.

| Component | Representation / model requirement |
|---|---|
| `hidden_states` | Decoder-layer output; separate hidden and residual outputs are combined for the intervention. |
| `attention_heads` | Concatenated query-head outputs before the output projection; standard decoder MHA/GQA with one GPU worker. |
| `router_logits` | MoE gate scores; the model must expose a gate output that executes during inference. |

| Algorithm | Component | Payload | `normalize=True` | `in_graph` |
|---|---|---|---|---|
| `direct` | `hidden_states` | `DirectionVector` | Yes | Yes |
| `attention_add` | `attention_heads` | `DirectionVector` | No | Yes |
| `erase` | `hidden_states` | `DirectionVector` | Yes | Yes |
| `replace` | `hidden_states` | `DirectionVector` | Yes | Yes |
| `concept_replace` | `hidden_states` | `ConceptPair` | Yes | Yes |
| `linear` | `hidden_states` | `LinearMap` | No | No |
| `loreft` | `hidden_states` | `ReftIntervention` | No | Rank at most `steer_graph_max_rank` |
| `lm_steer` | `hidden_states` | `LowRankProjector` | No | Rank at most `steer_graph_max_rank` |
| `moe_router` | `router_logits` | `RouterConfig` | No | `activate`, `deactivate`, `soft`, `soft_topk`; `soft_random` uses `split` |

MoE routing currently accepts only single-vector specs, including in `split`
and eager execution. For attention layout and unsupported attention types,
see the [attention guide](../user-guide/attention.md).

## Transformations and scale

`scale` is interpreted by the selected algorithm:

| Algorithm | Effect on selected rows |
|---|---|
| `direct`, `attention_add` | Add `scale * direction`. |
| `erase` | Remove the projection along the direction. Nonzero direction scaling largely cancels in the projection, apart from numerical stabilization. |
| `replace` | Replace the row with `scale * direction`. |
| `concept_replace` | Replace the component along `h1` with the same projection coefficient times `h2`; this algorithm does not use `scale`. |
| `linear` | Apply the affine map, then scale its output. |
| `loreft`, `lm_steer` | Add a learned low-rank update multiplied by `scale`. |
| `moe_router` | Apply the payload's routing mode and parameters; use its `lambda` parameter for soft routing strength. |

For a baseline, use `steering=False`. This unambiguously disables the entire
intervention, independently of the algorithm's scale semantics or engine default.

For supported algorithms, `normalize=True` rescales the transformed row to the
original row norm. It is separate from normalizing a direction during extraction.

## Native sources and checkpoint adapters

Use `VectorSpec(source=...)` for an EasySteer-defined format. Other checkpoint
schemas are loaded explicitly in the client and passed as `data`:

| Format | Input layout | Canonical payload | How to use it |
|---|---|---|---|
| `gguf` | EasySteer GGUF tensors named `direction.<layer>` | `DirectionVector` | `source="vector.gguf"` or `load(path, format="gguf")` |
| `concept_pair` | Directory with `h1.gguf` and `h2.gguf` | `ConceptPair` | `source="concepts/"` with `algorithm="concept_replace"` |
| `moe_router` | JSON with per-layer `layer_configs` | `RouterConfig` | `source="router.json"` with `algorithm="moe_router"` |
| `pt_direction` | One tensor or NumPy array saved with `torch.save` | `DirectionVector` | `load(path, format="pt_direction", layers=[10])` |
| `pyreft` | One intervention `.bin` and its config in a directory | `DirectionVector` for bias, `ReftIntervention` for LoReFT | `load(path, format="pyreft")` |
| `lm_steer` | Published projector checkpoint layout | `LowRankProjector` | `load(path, format="lm_steer", vector_index=0)` |
| `linear_transport` | Pickled transport with `A_` and optional `B_` | `LinearMap` | `load(path, format="linear_transport")` |

A file suffix such as `.pt` does not identify the checkpoint schema. Choose
`format` explicitly. A direction GGUF is also distinct from a quantized model
GGUF: it contains steering tensors, not model weights.

```python
from easysteer import vectors
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

payload = vectors.load("checkpoints/loreft", format="pyreft")
spec = SteeringSpec(vectors=[VectorSpec(
    data=payload,
    algorithm="loreft",
    scale=1.0,
    apply=ApplySpec(prompt_positions=[-1], generation="all"),
)])
```

The pyreft adapter preserves the checkpoint's layer index. In general, when a
payload records layer IDs, `VectorSpec.layers` restricts that set; it does not
relocate the weights. A payload without layer IDs, such as a `LinearMap` or
`LowRankProjector`, requires an explicit target-layer list.

`source` paths are resolved by the engine, so an HTTP client uses paths on the
server. Native loaders also accept `org/repo/file` Hugging Face sources when no
local path matches. For a reproducible deployment, download a chosen revision
first and use its local path.

## Using tensors directly

```python
import numpy as np
from vllm.steer_vectors import DirectionVector

# Use a direction learned for the model's layer 10.
direction = np.asarray(learned_direction, dtype=np.float32)
payload = DirectionVector({10: direction})
```

The width must match the selected model component. For attention, use the
capture layout rather than the hidden-state width. Both NumPy and PyTorch
tensors are accepted. The [payload class reference](payloads.md) gives constructor
arguments and shapes; tensor payloads validate dimensions and finite values
before workers materialize their tensors.

An extraction result can be converted without writing a temporary file:

```python
payload = vectors.from_control_vector(control_vector)
```

Native files and in-memory payloads share validation and content identity.
Requests receive a snapshot at admission; modifying the file or changing the
engine default does not change already-admitted requests. Identical payloads
can share a resident copy even when their request scales or selectors differ.

## JSON and HTTP payloads

Use the client helper to encode tensor bytes for JSON:

```python
from easysteer.vectors import to_json_payload

steering = {
    "vectors": [{
        "data": to_json_payload(payload),
        "algorithm": "direct",
        "apply": {"prompt": "all", "generation": "all"},
    }],
}
```

Here `payload` is a direction, as in the preceding tensor example. The helper
base64-encodes the canonical wire representation; HTTP consumers do not need
to invent a separate tensor schema. See [serving](../user-guide/openai-server.md)
for SDK requests and file preloading, and the
[adapter API](steer.md#payload-adapters-easysteervectors) for full signatures.
