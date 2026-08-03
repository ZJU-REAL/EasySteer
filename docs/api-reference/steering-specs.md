# Steering specs (`vllm.steer_vectors`)

The user-facing v2 steering API is defined in
[`vllm-steer/vllm/steer_vectors/api.py`](https://github.com/ZJU-REAL/EasySteer/blob/main/vllm-steer/vllm/steer_vectors/api.py)
and imported as:

```python
from vllm.steer_vectors import ApplySpec, SelectSpec, SteeringSpec, VectorSpec
```

**The [Steering guide](../user-guide/steering.md) is the canonical reference for these
classes** — every field of `SteeringSpec`, `VectorSpec`, `ApplySpec`, and `SelectSpec`
is documented there, with defaults, semantics, and examples.

This page is intentionally a hand-written summary rather than a generated one:
rendering these classes with mkdocstrings would require importing the `vllm-steer` fork
(and its CUDA/torch stack) in the docs build environment, and the docs build is kept
dependency-light so it can run in CI with only `requirements-docs.txt`.

## The four classes in one breath

| Class | Role |
|---|---|
| `SelectSpec` | The shared *where-clause* language: `phases`, `tokens`, `positions`, `exclude_tokens`, `exclude_positions`, `generation_window`. Also used by [hidden-state capture](../user-guide/hidden-state-capture.md). |
| `ApplySpec` | A `SelectSpec` subclass: where/when one vector applies. |
| `VectorSpec` | One vector: `source`, `algorithm`, `scale`, `layers`, `normalize`, `apply`, `params`, `name`. |
| `SteeringSpec` | Ordered `vectors` list plus a `conflict` policy (`"priority"` / `"sequential"` / `"error"`) and `debug`. |

For the design rationale, see the engineering record
[`STEERING_API_V2.md`](https://github.com/ZJU-REAL/EasySteer/blob/main/docs/design/STEERING_API_V2.md).
