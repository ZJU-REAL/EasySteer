# Contributing

Contributions are welcome in three main forms:

1. **Replications** — reproduce a steering paper as a notebook under `replications/`
   (README + notebook + vectors) and add it to the replications table.
2. **New steering algorithms** — subclass `AlgorithmTemplate` and register it.
3. **Component-level steers** — interfaces for steering attention/MLP modules are
   reserved in `vllm-steer/vllm/steer_vectors/models.py` and are a key focus of future
   updates.

## Adding a steering algorithm

An algorithm needs exactly two methods:

```python
import torch
from vllm.steer_vectors.algorithms.template import AlgorithmTemplate
from vllm.steer_vectors.algorithms.factory import register_algorithm

@register_algorithm("my_algorithm")
class MyAlgorithm(AlgorithmTemplate):

    def _transform(self, hidden_states: torch.Tensor, payload) -> torch.Tensor:
        """Apply the intervention. `payload` is one layer's entry from
        load_from_path's layer_payloads (Tensor or dict — your choice)."""
        return hidden_states + payload

    @classmethod
    def load_from_path(cls, path, device, *, config, target_layers=None, **kwargs):
        """Load per-layer payloads from a file (.gguf, .pt, ...).
        Returns {"layer_payloads": {layer_id: payload}}. Raise on
        underspecified inputs instead of assuming defaults."""
        if not target_layers:
            raise ValueError("my_algorithm requires target_layers")
        vector = torch.load(path, map_location=device, weights_only=False)
        return {"layer_payloads": {l: vector.to(config.adapter_dtype)
                                   for l in target_layers}}
```

Then export it from `vllm-steer/vllm/steer_vectors/algorithms/__init__.py`. The shared
template handles triggers (where-clauses), scaling, and normalization; your algorithm
only defines the math and the file format.

## Module map (`vllm-steer/vllm/steer_vectors/`)

| File | Role |
|---|---|
| `api.py` | User-facing v2 API (`SteeringSpec`/`VectorSpec`/`ApplySpec`) |
| `request.py` | Internal engine request struct + field registry |
| `worker_manager.py` | Config slots, fingerprints, vector store owner |
| `store.py` | Versioned vector store (dedup + reload) |
| `models.py` | Controller discovery & vector loading |
| `layers.py` | Slot-routed steering controllers (decoder/MoE gate) |
| `ops.py` | `vllm::steer_apply` custom op (piecewise graphs) |
| `trace.py` | Steering trace (test/debug oracle) |
| `algorithms/` | Algorithm framework & implementations |

## Ground rules

- Run the relevant [test suites](testing.md) before submitting.
- Fail explicitly: no silent defaults, no blanket `except` wraps.
- New behavior needs a test at the cheapest level that catches it (unit over e2e).

<!-- TODO: code style/linting instructions for the easysteer package itself
(the vllm-steer fork follows upstream vLLM's pre-commit setup). -->
