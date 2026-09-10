# Algorithm extension API

An algorithm transforms selected rows of one model component. The controller
handles hooks, row selection, request slots, and conflict resolution. Payload
loading and validation happen before execution. The
[contribution guide](../developer-guide/contributing.md#adding-a-steering-algorithm)
shows the registration and capability declaration needed to add an algorithm.

```python
from vllm.model_hooks.steering.algorithms.base import BaseSteerVectorAlgorithm
from vllm.model_hooks.steering.algorithms.registry import register_algorithm
```

## Transformation and graph contract

| Member | Extension contract |
|---|---|
| `_transform(hidden_state, params)` | Required. Return transformed selected rows with the same shape. `params` is one layer's prepared payload. |
| `set_payload(payload, scale_factor=1.0)` | Shared preparation: scale tensor payloads or attach `scale_factor` to a dictionary without mutating cached source tensors. |
| `_renormalize(original, transformed)` | Optional helper for algorithms declaring normalization support. Returns transformed rows rescaled to the original norms. |
| `graph_family` | `None` uses eager/split execution. A supported family declares that the algorithm can lower to persistent in-graph tables. |
| `graph_lower(payload, scale)` | Required for an in-graph family. Map the unscaled layer payload and request scale to that family's table keys. Its math must match `_transform`. |
| `graph_payload_problem(request=None)` | Return a restriction/rejection reason, or `None` when admissible. With no request, describe conditions that cannot be guaranteed from the algorithm name. |
| `wire_rank(wire)` | Low-rank families report payload rank so admission can enforce the engine's rank capacity. |

The eager preparation and graph-lowering inputs differ: tensor payloads passed
to `_transform` have already been scaled, while `graph_lower` receives the scale
separately. Handle scaling once in each path. For dictionary payloads, the
algorithm reads the prepared `scale_factor` as appropriate to its transformation.

`normalize` is a constructor option, not an automatic postprocessing step for
every algorithm. The algorithm must implement it, and its capability declaration
must permit it. For components whose decoder output is split into hidden and
residual tensors, the shared controller presents the complete selected rows to
the transformation and writes the resulting change back in the model's format.

::: model_hooks.steering.algorithms.base.BaseSteerVectorAlgorithm
    options:
      heading: BaseSteerVectorAlgorithm
      merge_init_into_class: true
      members: [graph_family, graph_payload_problem, graph_lower, wire_rank, set_payload, _transform, _renormalize]
      show_if_no_docstring: true

## Registration and construction

The registered name is used by `VectorSpec.algorithm` and the engine's
`steer_algorithms` declaration. Import the implementation in the algorithms
package so every engine process registers it.

::: model_hooks.steering.algorithms.registry.register_algorithm

::: model_hooks.steering.algorithms.registry.get_algorithm

::: model_hooks.steering.algorithms.registry.create_algorithm

`register_algorithm(name)` returns a class decorator and rejects duplicate
names. `get_algorithm(name)` returns the class; `create_algorithm(name,
normalize=False)` constructs it. Neither function loads weights.

## Authoring capabilities

The entry in `ALGORITHM_CAPABILITIES` defines accepted authoring inputs and
the component an algorithm edits. The runtime class above owns graph behavior.
After changing a capability, regenerate the packaged client contracts with
`python tools/export_client_contracts.py`.

::: model_hooks.steering.capabilities.AlgorithmCapabilities
    options:
      heading: AlgorithmCapabilities
      members: [payload_kind, source, normalize, params, target_component]
      show_if_no_docstring: true

`payload_kind` identifies a canonical [payload](payloads.md); `source` declares
native-source support; `normalize` and `params` constrain request fields;
`target_component` selects `hidden_states`, `attention_heads`, or `router_logits`.
See the [algorithm table](algorithms.md#capability-table) for current entries.
