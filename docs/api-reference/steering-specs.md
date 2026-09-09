# Steering specs (`vllm.steer_vectors`)

The user-facing v2 steering API is exported by
[`vllm-steer/vllm/steer_vectors/api.py`](https://github.com/ZJU-REAL/EasySteer-vllm-v1/blob/main/vllm/steer_vectors/api.py)
and imported as:

```python
from vllm.steer_vectors import ApplySpec, SelectSpec, SteeringSpec, VectorSpec
```

**The [Steering guide](../user-guide/steering.md) is the canonical reference for these
classes** — every field of `SteeringSpec`, `VectorSpec`, `ApplySpec`, and `SelectSpec`
is documented there, with defaults, semantics, and examples.

This page summarizes the fork's API by hand. The docs build analyzes the
`easysteer` package statically and does not load the vLLM fork; changes to these
specs should update this summary and the steering guide together.

## The four classes in one breath

| Class | Role |
|---|---|
| `SelectSpec` | The shared *where-clause* language: per-phase `prompt`/`generation` `"all"`, the six phase-scoped include selectors and their `exclude_*` twins. Also used by [hidden-state capture](../user-guide/hidden-state-capture.md). |
| `ApplySpec` | A `SelectSpec` subclass: where/when one vector applies. |
| `VectorSpec` | One vector: `source` or `data`, `algorithm`, `scale`, `layers`, `normalize`, `apply`, `params`, `name`. |
| `SteeringSpec` | Ordered `vectors` list plus a `conflict` policy (`"priority"` / `"sequential"` / `"error"`). |

The algorithm determines the target component: `moe_router` operates on
`router_logits`; the other algorithms operate on decoder `hidden_states`.
No separate target field is needed in `VectorSpec`. Steering and capture share
component discovery and availability checks.

Requests resolve `steering=None` (or omission) to the current default, `False` to
no steering, and a spec to a complete override. Python batches can provide these
choices per prompt. `LLM.set_default_steering(spec | None)` and
`get_default_steering()` manage the default; updates affect only new requests.

File and in-memory inputs become canonical payloads before execution:
`DirectionVector`, `LinearMap`, `LowRankProjector`, `ReftIntervention`,
`ConceptPair`, or `RouterConfig`. Explicit third-party format adapters are
available through `easysteer.vectors.load(path, format=...)`.

Graph eligibility also follows the algorithm and payload. In particular,
file-backed and inline `moe_router` configurations support `in_graph` for
`activate`, `deactivate`, `soft`, and `soft_topk`; `soft_random` requires `split`.
See [engine interactions](../user-guide/steering.md#interaction-with-engine-features)
for declaration-based auto selection and other payload conditions.

For engine integration, see the
[steering architecture](https://github.com/ZJU-REAL/EasySteer-vllm-v1/blob/main/docs/design/steer_vectors.md).
