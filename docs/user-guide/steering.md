# Steering requests

Steering is configured with three objects from `vllm.steer_vectors`:

1. **`ApplySpec`** — *where and when* a vector applies (phases, token/position filters,
   generation window).
2. **`VectorSpec`** — one vector: source file, algorithm, scale, layers, normalize,
   algorithm-specific `params`, and its `apply` clause.
3. **`SteeringSpec`** — an ordered list of `VectorSpec`s plus a conflict policy.

Eager, `split`, and `in_graph` engines share the same spec API, with the graph
payload conditions described below. A request using an algorithm outside the engine's declared `steer_algorithms`
is rejected at admission. Set `steer_algorithms=["direct"]` for direct steering,
or `steer_algorithms="all"` to serve all registered algorithms in `split` mode.
An engine-default `steering_config` can infer the declaration from its spec.
Requests with multiple vectors require `steer_multi_vector=True`
(`--steer-multi-vector` on the CLI), which is also implied by an `"all"`
declaration or a multi-vector engine-default spec.

## Attaching a spec

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

## `ApplySpec`: the where-clause

`ApplySpec` shares its selection language with hidden-state capture (both subclass
`SelectSpec`), so a clause means the same thing in both systems.

Each phase is selected independently and only by what you name: there is no
separate phase gate. `prompt="all"` selects every prompt token and
`generation="all"` every decode step — the widest include selector of each
phase — and six narrower include selectors, three per phase, each named for
it and carrying a symmetric exclude twin, refine the selection. A phase with
neither `"all"` nor a selector is untouched:

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
nothing is rejected — as is the removed `phases` key. One clause can mix
granularities across phases:

```python
# Last prompt token plus the whole generation (the SHARP shape):
ApplySpec(prompt_positions=[-1], generation="all")

# Prompt tail plus the first decode steps:
ApplySpec(prompt_window=(-4, None), generation_window=(0, 4))
```

## `VectorSpec`: one vector

| Field | Default | Meaning |
|---|---|---|
| `source` | `None` | EasySteer direction GGUF, a concept-pair directory, or `moe_router` JSON. For third-party checkpoint formats use `data` instead. Plain path only — no `"path\|algo"`. |
| `data` | `None` | An in-memory payload (see [Steering with your own tensors](#steering-with-your-own-tensors)). Mutually exclusive with `source`. |
| `algorithm` | `"direct"` | Registry key: `direct`, `attention_add`, `linear`, `loreft`, `lm_steer`, `erase`, `replace`, `concept_replace`, `moe_router`. |
| `scale` | `1.0` | Algorithm-specific scale factor; `direct` and `attention_add` add `scale * vector`. |
| `layers` | `None` | Layer indices to apply to; `None` uses the layer IDs in the source or payload. Payloads without layer IDs require an explicit list. |
| `normalize` | `False` | Rescale the transformed hidden state to its original norm for `direct`, `erase`, `replace`, and `concept_replace`. Other algorithms reject `True`. |
| `apply` | — | **Required** `ApplySpec`. |
| `params` | `{}` | Algorithm-specific parameters, validated per algorithm; unknown keys are rejected. Only `moe_router` takes params: `expert_ids`, `mode`, `lambda`, `topk`. |
| `name` | `None` | Label used in logs only (not identity). |

For `moe_router`, explicit `mode`, `lambda` and `topk` parameters override those
values in a source file or `RouterConfig`; omitted parameters retain the payload's
values. `expert_ids` in `params` is only for inline configuration without `source`
or `data`. A file or `RouterConfig` records its own expert IDs per layer.

## Steering with your own tensors

The engine loads only formats whose schema EasySteer defines. Everything else —
pyreft checkpoints, LM-Steer `.pt` files, pickled transport maps, or tensors you
just computed — is passed in memory through `VectorSpec(data=...)` using the
canonical payload structures:

| Payload | Algorithms | Shape |
|---|---|---|
| `DirectionVector({layer: vec})` | `direct`, `attention_add`, `erase`, `replace` | one 1-D vector per layer; `attention_add` uses concatenated query head outputs |
| `LinearMap(weight, bias=None)` | `linear` | one affine map, applied to each `layers` entry |
| `LowRankProjector(projector1, projector2)` | `lm_steer` | low-rank update factors, applied to each `layers` entry |
| `ReftIntervention(rotate_layer, learned_source_weight, learned_source_bias=None, layer=None)` | `loreft` | Uses its recorded `layer`; without one, requires `VectorSpec.layers` |
| `ConceptPair(h1=..., h2=...)` | `concept_replace` | replace the component along `h1` with the corresponding component along `h2` |
| `RouterConfig({layer: config})` | `moe_router` | per-layer expert IDs and routing mode |

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

When a payload records layer IDs, `VectorSpec.layers` restricts those IDs; it does
not move an intervention to a different layer. For a checkpoint with a recorded
LoReFT layer, omit `layers` or include that layer in the list.

## `SteeringSpec`: vectors + conflict policy

| Field | Default | Meaning |
|---|---|---|
| `vectors` | — | Non-empty ordered list of `VectorSpec`s. |
| `conflict` | `"priority"` | When several vectors target one position: `"priority"` (first wins), `"sequential"` (stack in order), `"error"`. |

`moe_router` is not yet supported in multi-vector specs.

## Examples

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

- **Prefix caching** is supported: block hashes include the effective request's
  steering fingerprint. Default and explicit requests with identical configs can
  reuse blocks; different configs remain separate. Updating or clearing the default
  preserves reusable blocks for previous configurations.
- **Chunked prefill** is supported; negative positions resolve stably across chunks.
- **Beam search** requires steering to be disabled. Use `steering=False` (HTTP:
  `"steering": false`) when a default is configured; effective steering with beam
  search is rejected because its prompt/generation selection semantics differ.
- **CUDA graphs**: `split` supports all algorithms and multi-vector specs, with
  steering between compiled graph segments. `in_graph` requires a single-vector
  spec whose algorithm has a graph kernel: `direct`, `attention_add`, `erase`,
  `replace`, and `concept_replace` support it; `loreft` and `lm_steer` also require
  payload rank at most `steer_graph_max_rank` (default 32). `moe_router` supports inline and
  file-backed `activate`, `deactivate`, `soft`, and `soft_topk` configurations;
  `soft_random` and `linear` use `split`.
  Normalization for `direct`, `erase`, `replace`, and `concept_replace` is supported
  in both tiers; `attention_add` requires `normalize=False`. The default
  `steer_graph_mode="auto"` evaluates an engine-default spec's actual payloads;
  with a names-only declaration, conditional algorithms select `split`.
  Resolution and admission share the same capability rules, and the engine logs
  its selection reason. Eager execution uses
  the same specs without graph capture. See the
  [engine guide](https://github.com/ZJU-REAL/EasySteer-vllm-v1/blob/main/docs/features/steer_vectors.md#graph-tiers)
  for explicit graph settings.

The algorithm selects its model component: `moe_router` edits `router_logits`
at an accessible MoE gate; `attention_add` edits `attention_heads` before the
attention output projection. The other algorithms edit decoder `hidden_states`.
Steering and capture share layer discovery and hook availability, including
checks for fused routers whose gate modules are bypassed.

See [graphs, caching, and performance](performance.md) for startup selection,
capture execution, and tuning. The [algorithm reference](../api-reference/algorithms.md)
compares payload formats, scale semantics, and graph conditions in one table.

## Attention head intervention

`attention_add` adds a direction to the concatenated attention head outputs,
after attention aggregation and before the output projection. Declare
`steer_algorithms=["attention_add"]` when creating the engine. It supports eager,
`split`, and `in_graph` execution; the default `auto` mode selects `in_graph`
for a single-vector attention declaration.

```python
attention = SteeringSpec(vectors=[VectorSpec(
    source=".runtime/iti/iti.gguf",
    algorithm="attention_add",
    scale=3.0,
    apply=ApplySpec(prompt_positions=[-1], generation="all"),
)])
llm.generate(prompts, steering=attention)
```

Use a standard decoder MHA/GQA model with `tensor_parallel_size=1`.
MLA, encoder attention, and cross-attention are outside this component's scope.
Each layer's vector width is `num_heads * head_size`, obtained from the
[capture layout](hidden-state-capture.md#attention-head-outputs), and need not
equal the residual hidden size. `num_heads` counts query heads, including for
GQA. To select heads, leave the other head slices zero in the existing
`DirectionVector` or GGUF payload. `normalize=True` is rejected.

The [ITI example](extracting-vectors.md#iti-attention-head-directions) learns
head directions and their scale from TruthfulQA development data. Its
`ApplySpec` includes the last prompt token so the first answer-token prediction
is also steered.
The [attention guide](attention.md) explains the component layout, captures its
head outputs, and links the complete paper replication.

## Migrating from v1

The v1 surface (trigger fields, `steer_vector_request`, `--steer-vector-path` flags, the
`-1` token sentinel, `"path|algo"` sources) has been **deleted**. Key semantic changes:

- Exclusions always subtract; nothing bypasses them.
- `generation_window=(0, k)` steers exactly `k` decode steps (the v1 `first_k`
  off-by-one is gone).
- `prompt="all"` / `generation="all"` replace the `-1` token sentinel. There is
  no `phases` field; narrower selectors imply their own phase.
- `normalize` defaults to `False` everywhere, including default configurations.
- `generation_window` is an include selector like any other: it **unions** with
  the token/position selectors instead of constraining decode tokens, and a
  cross-phase clause with only a `generation_window` no longer covers the
  prompt — select prompt tokens explicitly (e.g. `prompt_window=(0, None)`).
- Every selector is phase-scoped and named for it: token-id filters split into
  `prompt_tokens` / `generation_tokens`, and `prompt_positions` (formerly
  `positions`) selects prompt tokens only — decode steps are always addressed
  through `generation_positions` / `generation_window`.
