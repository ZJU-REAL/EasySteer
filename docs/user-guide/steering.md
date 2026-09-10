# Steering requests

Use a steering request to select a payload, its strength, and the model rows to
change. This guide covers request overrides, token selection, and combinations
of vectors. For a complete runnable example, start with the
[quickstart](../getting-started/quickstart.md).

## Requirements

Use the EasySteer vLLM fork with steering enabled. Declare the algorithms the
engine will serve with `steer_algorithms`, or provide a startup
`steering_config` from which they can be inferred. Requests must fit that
[engine declaration](../api-reference/engine-configuration.md).

A request containing multiple vectors also requires `steer_multi_vector=True`.
An `"all"` algorithm declaration or a multi-vector startup default implies this
setting. The model must expose the component used by the chosen algorithm;
see the [capability table](../api-reference/algorithms.md#capability-table).

## Build a configuration

Import the three authoring classes from `vllm.steer_vectors`:

| Class | Configure | Reference |
|---|---|---|
| `VectorSpec` | One payload, algorithm, scale, layer set, and application rule. | [Fields and defaults](../api-reference/steering-specs.md#model_hooks.steering.api.VectorSpec) |
| `ApplySpec` | Which prompt and generation rows the vector changes. | [Selection fields](../api-reference/steering-specs.md#model_hooks.selection.spec.SelectSpec) |
| `SteeringSpec` | An ordered list of vectors and a conflict policy. | [Fields and defaults](../api-reference/steering-specs.md#model_hooks.steering.api.SteeringSpec) |

## Set steering for a request {#attaching-a-spec}

Every request resolves one effective configuration at admission. Python and HTTP
use the same rules:

| `steering` value | Effective configuration |
|---|---|
| Omitted or `None` (`null` in JSON) | The current default, if one exists. |
| `False` (`false` in JSON) | No steering for this request. |
| A `SteeringSpec` | This complete spec, overriding the default. |

`llm.generate(prompts, steering=spec, ...)` applies one spec to the batch. A list
such as `steering=[None, False, spec]` resolves these choices separately for each
prompt. Defaults and overrides are not stacked.

Set an initial default with `LLM(steering_config=...)` or
`--steering-config spec.json` (inline JSON is also accepted). Update or clear it
with `llm.set_default_steering(spec)` or `llm.set_default_steering(None)`;
`llm.get_default_steering()` reports the current default. HTTP clients use
`POST /v1/steering {"spec": ...}` or `{"spec": null}` and `GET /v1/steering`.
See [OpenAI-compatible server](openai-server.md).

Updates affect new requests. Requests already admitted retain their configuration
and weight snapshot. Changing a default does not reset the prefix cache or reserve
a permanent execution slot. Algorithms, graph mode and capacity remain engine
settings, so every default and override must satisfy that declaration.

## Select prompt and generation rows

`ApplySpec` inherits the `SelectSpec` language used by capture. Select prompt
rows and generation rows independently. `prompt="all"` selects the full prompt;
`generation="all"` selects every decode step. Use the fields below for a subset.
A phase with neither `"all"` nor an include selector is unchanged.

| Include | Exclude twin | Matches |
|---|---|---|
| `prompt_tokens` | `exclude_prompt_tokens` | Token-id allowlist (real ids, `>= 0`) over prompt occurrences. |
| `prompt_positions` | `exclude_prompt_positions` | Prompt positions; negative values are Python-style from the end of the prompt (`-1` = last prompt token), stable across prefill chunks. Positive values past the prompt end clamp to the last prompt token (warned at admission). |
| `prompt_window` | `exclude_prompt_window` | Half-open `(start, stop)` over prompt positions; negative bounds and `stop=None` resolve from the prompt end (`(-5, None)` = the last five prompt tokens). |
| `generation_tokens` | `exclude_generation_tokens` | Token-id allowlist over generated occurrences. |
| `generation_positions` | `exclude_generation_positions` | Exact 0-based decode steps (`[0]` = the first generated token). |
| `generation_window` | `exclude_generation_window` | Half-open `(start, stop)` over 0-based decode steps; `stop=None` = unbounded. `(0, k)` selects exactly the first `k` decode steps. |

The include selectors (with `prompt="all"` / `generation="all"` among them)
select the **union** of their matches. The exclude selectors union and
**always subtract**: where an include and an exclude overlap, the exclusion
wins; an exclude requires its phase to be covered, and a clause that selects
nothing is rejected. One clause can mix granularities across phases:

```python
# Last prompt token plus the whole generation (the SHARP shape):
ApplySpec(prompt_positions=[-1], generation="all")

# Prompt tail plus the first decode steps:
ApplySpec(prompt_window=(-4, None), generation_window=(0, 4))
```

## Choose weights, layers, and strength

Provide weights through `source` or `data`; these fields are mutually exclusive.
Inline `moe_router` configuration can use `params` alone, as described below.

- Use `source` for an EasySteer direction GGUF, concept-pair directory, or router
  JSON. The engine resolves the path; HTTP clients use a path on the server.
- Use `data` for a [payload object](../api-reference/payloads.md), including
  tensors constructed in Python or loaded through a checkpoint adapter.

`layers` selects true decoder-layer IDs. If the payload records its own IDs,
`layers` restricts that set; it does not move weights to a different layer.
Payloads without recorded IDs require an explicit list.

For `direct` and `attention_add`, `scale` multiplies the added direction.
Other algorithms have their own [scale semantics](../api-reference/algorithms.md#transformations-and-scale).
Use `steering=False` for a baseline. `normalize=True` preserves the original row
norm only for algorithms that support it; it is separate from normalizing a
vector during extraction. Every vector requires an explicit `apply` selection.

The [VectorSpec reference](../api-reference/steering-specs.md#model_hooks.steering.api.VectorSpec)
provides all field types, defaults, and validation rules. Only `moe_router`
accepts algorithm-specific `params`.

For `moe_router`, explicit `mode`, `lambda` and `topk` parameters override those
values in a source file or `RouterConfig`; omitted parameters retain the payload's
values. `expert_ids` in `params` is only for inline configuration without `source`
or `data`. A file or `RouterConfig` records its own expert IDs per layer.

## Steering with your own tensors

Load third-party checkpoints in the client through
[`easysteer.vectors`](../api-reference/steer.md#payload-adapters-easysteervectors),
then pass the returned payload as `VectorSpec.data`. Use the
[format table](../api-reference/algorithms.md#native-sources-and-checkpoint-adapters)
to choose an adapter, or construct a [payload object](../api-reference/payloads.md)
from your own tensors.

This fragment assumes `my_vector` contains a direction for layer 10 of the
loaded model, with the same width as that layer's hidden states:

```python
from vllm.steer_vectors import ApplySpec, DirectionVector, SteeringSpec, VectorSpec

# Any tensor you have — numpy or torch — becomes steerable directly:
spec = SteeringSpec(vectors=[VectorSpec(
    data=DirectionVector({10: my_vector}),
    scale=2.0,
    apply=ApplySpec(prompt="all", generation="all"),
)])
```

Payloads are validated at construction (shapes, finiteness, role names) and
identified engine-side by a content hash, so identical payloads share one
resident copy regardless of how many requests carry them. Native files and
in-memory data use the same payload validation, layer selection, identity and
materialization path. The file is read before execution; workers consume that
snapshot rather than reopen the source.

`easysteer.vectors.load(path, format=...)` selects an explicit format adapter.
Supported formats are `gguf`, `concept_pair`, `moe_router`, `pt_direction`,
`pyreft`, `lm_steer`, and `linear_transport`. The existing convenience adapters
remain available (`from_pyreft`,
`from_lm_steer`, `from_linear_transport`, `from_pt_direction`, `from_gguf`),
and `easysteer.vectors.from_control_vector(cv)` steers an extraction result
with no GGUF round-trip.

For a checkpoint with a recorded LoReFT layer, omit `layers` or include that
layer in the list.

## Combine vectors

| Field | Default | Meaning |
|---|---|---|
| `vectors` | — | Non-empty ordered list of `VectorSpec`s. |
| `conflict` | `"priority"` | When several vectors target one position: `"priority"` (first wins), `"sequential"` (stack in order), `"error"`. |

`moe_router` is not yet supported in multi-vector specs.

### Selection and combination examples

```python
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

# Single vector on every prompt + generated token
sentiment = SteeringSpec(vectors=[
    VectorSpec(source="vectors/happy_diffmean.gguf", scale=2.0, layers=[10, 11, 12],
               apply=ApplySpec(prompt="all", generation="all")),
])

# Several directions stacked at the second-to-last prompt token.
# The engine must have steer_multi_vector=True; supply your own dir1/dir2 files.
multi = SteeringSpec(
    conflict="sequential",
    vectors=[
        VectorSpec(source="dir1.gguf", scale=1.5, layers=[20],
                   apply=ApplySpec(prompt_positions=[-2])),
        VectorSpec(source="dir2.gguf", scale=-0.8, layers=[20],
                   apply=ApplySpec(prompt_positions=[-2])),
    ],
)

# Steer only the first 8 generated tokens
early = SteeringSpec(vectors=[
    VectorSpec(source="vectors/happy_diffmean.gguf", scale=2.0, layers=[10, 11, 12],
               apply=ApplySpec(generation_window=(0, 8))),
])
```

## Interaction with engine features

| Feature | Steering behavior | Details |
|---|---|---|
| Prefix caching | Matching prompts and effective configurations can reuse blocks; different steering configurations remain separate. Updating a default preserves reusable entries. | [Cache identity](performance.md#prefix-caching) |
| Chunked prefill | Selection uses positions in the complete prompt, including negative positions. | [Selection rules](#select-prompt-and-generation-rows) |
| CUDA graphs | `split` serves all algorithms; `in_graph` serves single-vector requests that satisfy the algorithm's payload conditions. | [Mode selection](performance.md#graph-modes) · [Algorithm conditions](../api-reference/algorithms.md#capability-table) |
| Beam search | Effective steering must be disabled, including any engine default. Pass `steering=False`. | Beam search uses different prompt/generation selection semantics. |

The algorithm determines the component: `attention_add` edits `attention_heads`,
`moe_router` edits `router_logits`, and the other algorithms edit
`hidden_states`. Steering and capture use the same layer discovery and check
that the relevant hook runs, including for fused MoE implementations.

## Attention head intervention

To steer attention head outputs, declare `steer_algorithms=["attention_add"]`
and use a direction extracted for those outputs. The request uses the same
spec and selection language:

```python
attention = SteeringSpec(vectors=[VectorSpec(
    source=".runtime/iti/iti.gguf",
    algorithm="attention_add",
    scale=3.0,
    apply=ApplySpec(prompt_positions=[-1], generation="all"),
)])
llm.generate(prompts, steering=attention)
```

The [attention guide](attention.md) covers supported MHA/GQA models, head layout,
capture, and the ITI replication. Use its captured query-head layout to size the
vector; attention head outputs need not have the model's hidden-state width.

## Older steering APIs {#migrating-from-v1}

The earlier steering API used `steer_vector_request`, trigger fields,
`--steer-vector-path`, and `"path|algo"` sources. These interfaces have been
removed. Use `SteeringSpec`, plain `source` paths, and phase-specific selections.
This API change is separate from vLLM's V1/V2 model-runner naming.

When updating an old example, check selection semantics: includes form a union,
exclusions subtract from it, and a generation window selects only decode rows.
`generation_window=(0, k)` selects the first `k` decode steps. Select prompt rows
explicitly when needed; `prompt_positions` never selects generation rows.
