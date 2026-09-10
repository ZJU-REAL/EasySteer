# Steering specs (`vllm.steer_vectors`)

The classes below are the public authoring API. Their descriptions, field types,
defaults, and methods are rendered from the fork's source definitions.

```python
from vllm.steer_vectors import ApplySpec, SelectSpec, SteeringSpec, VectorSpec
```

| Class | Purpose |
|---|---|
| [SteeringSpec](#model_hooks.steering.api.SteeringSpec) | A complete request configuration: ordered vectors and a conflict policy. |
| [VectorSpec](#model_hooks.steering.api.VectorSpec) | One payload, algorithm, scale, layer set, and application rule. |
| [ApplySpec](#model_hooks.steering.api.ApplySpec) | Token selection for a steering vector. |
| [SelectSpec](#model_hooks.selection.spec.SelectSpec) | The same selection language, also exported by `vllm.capture`. |

For worked examples and selection semantics, see
[steering requests](../user-guide/steering.md). The
[algorithm reference](algorithms.md) covers accepted payloads and graph
conditions; [engine configuration](engine-configuration.md) covers startup
declarations and capacity.

## Constructing and validating specs

```python
spec = SteeringSpec(vectors=[VectorSpec(
    source="vectors/happy_diffmean.gguf",
    algorithm="direct",
    scale=2.0,
    layers=list(range(10, 24)),
    apply=ApplySpec(prompt="all", generation="all"),
)])

text = spec.model_dump_json()
restored = SteeringSpec.model_validate_json(text)
```

Specs are Pydantic models. Constructors and `model_validate(...)` validate field
values and reject unknown keys. `model_dump(mode="json")` returns a dictionary;
`model_dump_json()` returns JSON text; `model_validate_json(text)` returns a
validated model. For tensor payloads that must be sent as JSON, use
[`to_json_payload`](steer.md#easysteer.vectors.to_json_payload) before placing
the payload in `VectorSpec.data`.

Pass a complete spec to `llm.generate(..., steering=spec)`. Omission or `None`
inherits the current engine default; `False` disables steering. Python batches
can supply one choice per prompt. Defaults and explicit specs are not stacked.

::: model_hooks.steering.api.SteeringSpec
    options:
      heading: vllm.steer_vectors.SteeringSpec
      members: [vectors, conflict]
      show_if_no_docstring: true

::: model_hooks.steering.api.VectorSpec
    options:
      heading: vllm.steer_vectors.VectorSpec
      members: [source, data, algorithm, scale, layers, normalize, apply, params, name]
      show_if_no_docstring: true

::: model_hooks.steering.api.ApplySpec
    options:
      heading: vllm.steer_vectors.ApplySpec
      members: false

`ApplySpec` inherits every field, default, validation rule, and wire-conversion
method from `SelectSpec` below. It is the type required by `VectorSpec.apply`.

::: model_hooks.selection.spec.SelectSpec
    options:
      heading: vllm.capture.SelectSpec
      members:
        - prompt
        - generation
        - prompt_tokens
        - prompt_positions
        - prompt_window
        - generation_tokens
        - generation_positions
        - generation_window
        - exclude_prompt_tokens
        - exclude_prompt_positions
        - exclude_prompt_window
        - exclude_generation_tokens
        - exclude_generation_positions
        - exclude_generation_window
        - to_wire
        - from_wire
      show_if_no_docstring: true

## Engine-facing conversion

Applications normally pass a `SteeringSpec` directly to `LLM.generate()` or the
HTTP API. Integration code can use `to_engine_request()` to resolve files and
payloads into an internal `SteeringRequest`. Its `name` and `int_id` are labels;
payload content and application fields determine execution identity.

::: model_hooks.steering.api.to_engine_request
    options:
      heading: vllm.steer_vectors.to_engine_request

The returned request is an engine transport object, not an alternative authoring
schema. File loading can fail here before request admission.

## Engine default and preload methods

These are EasySteer's additions to `vllm.LLM`. For the rest of the class, refer
to the upstream [LLM API](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/).
Default changes affect new requests; admitted requests retain their snapshots.

::: llm.LLM.set_default_steering
    options:
      heading: LLM.set_default_steering

::: llm.LLM.get_default_steering
    options:
      heading: LLM.get_default_steering

::: llm.LLM.preload_steer_vectors
    options:
      heading: LLM.preload_steer_vectors

For remote clients, the corresponding operations are described under
[HTTP management endpoints](../user-guide/openai-server.md#management-endpoints).
