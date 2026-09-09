# Contributing

Contributions are welcome in three main forms:

1. **Replications** — reproduce a steering paper as a notebook under `replications/`
   (README + notebook + vectors) and add it to the replications table.
2. **New steering algorithms** — subclass `BaseSteerVectorAlgorithm` and register it.
3. **Model and component support** — extend module discovery and steering controllers,
   with tests for the model's hidden-state and residual layout.

## Adding a steering algorithm

An algorithm implements `_transform`, which receives the selected hidden-state rows
and one layer's canonical payload. Declare its payload kind in `capabilities.py`;
the shared loader handles supported native files, validation and materialization.
For example, this additive algorithm consumes the same `direction` payload as
`direct`, `erase`, and `replace`:

```python
import torch
from vllm.model_hooks.steering.algorithms.base import BaseSteerVectorAlgorithm
from vllm.model_hooks.steering.algorithms.registry import register_algorithm

@register_algorithm("my_algorithm")
class MyAlgorithm(BaseSteerVectorAlgorithm):

    def _transform(self, hidden_states: torch.Tensor, payload) -> torch.Tensor:
        # Tensor payloads have already been multiplied by the requested scale.
        transformed = hidden_states + payload
        if self.normalize:
            return self._renormalize(hidden_states, transformed)
        return transformed

```

Its entry in `capabilities.py` is
`"my_algorithm": AlgorithmCapabilities("direction", "gguf", normalize=True)`.
Algorithms sharing a payload kind also share its loader. Add a native format
reader to `loading.py` only when a new EasySteer schema is needed; third-party
checkpoint layouts belong in explicit `easysteer.vectors` adapters. File and
in-memory paths must produce equivalent canonical payloads.

Import the class from `vllm-steer/vllm/model_hooks/steering/algorithms/__init__.py` so every
engine process registers it, then include `"my_algorithm"` in `steer_algorithms`.
The controllers handle selection and conflict resolution; the base class prepares
scaled tensor payloads and provides `_renormalize`. Each algorithm decides how to
apply normalization in its transformation.

New algorithms run in eager or `split` mode by default. To support `in_graph`, also
declare a `graph_family` and implement `graph_lower` with matching math; low-rank
families need `wire_rank` for admission checks. Use the existing algorithm classes
and [engine architecture](https://github.com/ZJU-REAL/EasySteer-vllm-v1/blob/main/docs/design/steer_vectors.md)
as references. Test eager and graph implementations against the same inputs.

Declare the payload kind, source format, normalization support and allowed
`VectorSpec.params` in `capabilities.py`. Run `python tools/export_client_contracts.py`
from the repository root to update the packaged browser and HF client rules.
Provide a client-side checkpoint adapter when needed. Graph capabilities remain
on the algorithm class.

## Shared model hooks

The public Python APIs remain `vllm.steer_vectors` and `vllm.capture`.
Their implementation lives under `vllm-steer/vllm/model_hooks/`:

```text
model_hooks/
  selection/              # Selection specs, token matching and batch geometry
  components/             # Model discovery and component output adapters
  steering/
    algorithms/           # Transformations and their registration
    controllers/          # Hidden-state and router-logit execution
    graph/                # Eligibility, persistent state and kernels
  capture/                # Capture sessions, selection, storage and serialization
```

Capture and steering independently consume `selection` and `components`.
Capture can run without a steering declaration, algorithm or payload cache.
The shared packages have no dependency on either consumer.
`selection/spec.py` defines `SelectSpec`, `schema.py` defines its fields, and
`runtime.py` resolves token selections against batch geometry.

## Steering modules (`model_hooks/steering/`)

| File | Role |
|---|---|
| `api.py` | User-facing v2 API (`SteeringSpec`/`VectorSpec`/`ApplySpec`) |
| `request.py` | `SteeringRequest` with an ordered list of `ResolvedVector` payloads and application fields |
| `defaults.py` | Default configuration snapshots and per-request inheritance / override resolution |
| `input_validation.py` | Shared source, data and algorithm-parameter validation before file loading |
| `loading.py` | Native file adapters and content snapshots before request admission |
| `payloads.py` | Canonical payload validation, content identity and materialization |
| `capabilities.py` | Algorithm authoring rules shared with packaged clients |
| `worker_manager.py` | `WorkerSteeringState`: config slots, fingerprints and payload-cache ownership |
| `payload_cache.py` | `PayloadCache` of content-addressed, materialized per-layer payload dictionaries |
| `controllers/base.py` | Shared slot lifecycle and graph-buffer interface |
| `controllers/hidden_states.py` / `controllers/router_logits.py` | Component-specific steering, with eager and graph execution kept together |
| `controllers/manager.py` | Index controllers by component and layer; attach hooks and install/release slot payloads |
| `graph/policy.py` | Graph mode resolution and request admission |
| `graph/state.py` | Persistent graph tables, slot distribution and step masks |
| `graph/kernels.py` | Steering tensor kernels |
| `ops.py` | `vllm::steer_apply` custom op (piecewise graphs) |
| `trace.py` | Steering trace (test/debug oracle) |
| `algorithms/` | Algorithm framework & implementations |

Single-vector and multi-vector specs use the same internal request structure;
each resolved vector carries a canonical `payload` regardless of its input format.
The original `source` is retained for reporting and preload policy, while worker
caches use payload content and broadcast layer targets. HTTP management routes
live in `vllm/entrypoints/serve/steering/api_router.py`.

`components/registry.py` supplies shared component descriptors and hook targets;
`components/discovery.py` locates decoder layers and accessible MoE gates;
`components/outputs.py` handles component output layouts. Steering and capture
consume the same discovered component targets. Standard decoder
stacks use their global stack indices; pipeline-parallel placeholders retain
those indices. Architecture exceptions belong in discovery and output-layout
rules, rather than separate name parsing in each consumer. ReFT uses its own
Transformers model profiles; repeated decoder layouts share an implementation,
while component dimensions and special layouts remain explicit.

Controllers declare their graph mask names and allocate their own component
tables. Shared graph state calls that interface without branching on concrete
controller classes. A new component supplies its discovery, output adapter,
controller and graph-buffer contract; its algorithms still own their graph math.

Capture keeps its session, graph dispatch, selection, store and serialization
modules together under `model_hooks/capture/`. Its selection module attaches
capture row labels and reduction plans to the shared token-selection result.
`serialization.py` owns the dtype and wire-format contract in both directions;
`store.py` owns buffered values, row budgets and clearing.

## Ground rules

- Run the relevant [test suites](testing.md) before submitting.
- Fail explicitly: no silent defaults, no blanket `except` wraps.
- New behavior needs a test at the cheapest level that catches it (unit over e2e).
- Update the docs: the relevant page under `docs/`, and the README pointer line if a
  user-facing surface changed (the PR template has a checklist).

For the current public contracts, read the [steering guide](../user-guide/steering.md),
[capture guide](../user-guide/hidden-state-capture.md), and the engine architecture
linked above. The `vllm-steer` fork follows upstream vLLM's pre-commit configuration.
