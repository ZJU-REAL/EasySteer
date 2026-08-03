# Steering (v2 API)

Steering is configured with three objects from `vllm.steer_vectors`
(see [`STEERING_API_V2.md`](https://github.com/ZJU-REAL/EasySteer/blob/main/docs/design/STEERING_API_V2.md)
for the design rationale):

1. **`ApplySpec`** — *where and when* a vector applies (phases, token/position filters,
   generation window).
2. **`VectorSpec`** — one vector: source file, algorithm, scale, layers, normalize,
   algorithm-specific `params`, and its `apply` clause.
3. **`SteeringSpec`** — an ordered list of `VectorSpec`s plus a conflict policy.

Specs are backend-independent: eager, piecewise, and full CUDA-graph engines accept the
same spec; a backend that cannot run a spec rejects it at admission.

## Attaching a spec

Same object, two scopes:

- **Per request**: `llm.generate(prompts, steering=spec, ...)`, or the JSON field
  `"steering"` on HTTP requests (see [OpenAI-compatible server](openai-server.md)).
- **Engine default**: `--steering-config spec.json` (or inline JSON) at startup,
  replaceable at runtime via `POST /v1/steering {"spec": {...}}` (resets the prefix
  cache).

Server-level and per-request steering cannot currently be combined in one request.

## `ApplySpec`: the where-clause

`ApplySpec` shares its selection language with hidden-state capture (both subclass
`SelectSpec`), so a clause means the same thing in both systems.

| Field | Meaning |
|---|---|
| `phases` | **Required, non-empty.** Which token kinds to select: `"prompt"`, `"generation"`. With no other filters, selects every token of the listed phases. |
| `tokens` | Token-id allowlist (real ids, `>= 0`). |
| `positions` | Absolute sequence positions; negative values are Python-style from the end of the *prompt* (`-1` = last prompt token), stable across prefill chunks. |
| `exclude_tokens` / `exclude_positions` | Always subtract, in every case. |
| `generation_window` | Half-open `(start, stop)` over 0-based decode steps; `stop=None` = unbounded. `(0, k)` steers exactly the first `k` decode steps. Requires `"generation"` in `phases`. |

Within the selected phases, `tokens` and `positions` select the **union** of their
matches; exclusions always subtract.

## `VectorSpec`: one vector

| Field | Default | Meaning |
|---|---|---|
| `source` | `None` | Path to the vector file. Required for every algorithm except `moe_router` with explicit `expert_ids`. Plain path only — no `"path\|algo"`. |
| `algorithm` | `"direct"` | Registry key: `direct`, `linear`, `loreft`, `lm_steer`, `erase`, `replace`, `concept_replace`, `moe_router`. |
| `scale` | `1.0` | Scale factor (negative suppresses the direction). |
| `layers` | `None` | Layer indices to apply to; `None` lets the file decide. |
| `normalize` | `False` | Normalize the vector before applying. |
| `apply` | — | **Required** `ApplySpec`. |
| `params` | `{}` | Algorithm-specific parameters, validated per algorithm; unknown keys are rejected. Only `moe_router` takes params: `expert_ids`, `mode`, `lambda`, `topk`. |
| `name` | `None` | Label used in logs only (not identity). |

## `SteeringSpec`: vectors + conflict policy

| Field | Default | Meaning |
|---|---|---|
| `vectors` | — | Non-empty ordered list of `VectorSpec`s. |
| `conflict` | `"priority"` | When several vectors target one position: `"priority"` (first wins), `"sequential"` (stack in order), `"error"`. |
| `debug` | `False` | Verbose logging during the forward pass. |

`moe_router` is not yet supported in multi-vector specs.

## Examples

```python
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

# Single vector on every prompt + generated token
sentiment = SteeringSpec(vectors=[
    VectorSpec(source="vectors/happy.gguf", scale=2.0, layers=[10, 11, 12],
               apply=ApplySpec(phases=["prompt", "generation"])),
])

# Several directions stacked at the second-to-last prompt token
multi = SteeringSpec(
    conflict="sequential",
    vectors=[
        VectorSpec(source="dir1.gguf", scale=1.5, layers=[20],
                   apply=ApplySpec(phases=["prompt"], positions=[-2])),
        VectorSpec(source="dir2.gguf", scale=-0.8, layers=[20],
                   apply=ApplySpec(phases=["prompt"], positions=[-2])),
    ],
)

# Steer only the first 8 generated tokens
early = SteeringSpec(vectors=[
    VectorSpec(source="vectors/happy.gguf", scale=2.0, layers=[10, 11, 12],
               apply=ApplySpec(phases=["generation"], generation_window=(0, 8))),
])
```

## Interaction with engine features

- **Prefix caching** is supported: block hashes are keyed by the steering config
  fingerprint; engine-default mode salts every hash, and spec updates reset the cache.
- **Chunked prefill** is supported; negative positions resolve stably across chunks.
- **CUDA graphs**: full-graph mode currently requires a single-vector, `direct`,
  non-normalized spec; other specs are rejected at admission with an explicit error.
  `--steer-graph-mode` is an engine optimization setting and never changes how a spec is
  written.

## Migrating from v1

The v1 surface (trigger fields, `steer_vector_request`, `--steer-vector-path` flags, the
`-1` token sentinel, `"path|algo"` sources) has been **deleted**. Key semantic changes:

- Exclusions always subtract; nothing bypasses them.
- `generation_window=(0, k)` steers exactly `k` decode steps (the v1 `first_k`
  off-by-one is gone).
- Phase selection (`phases`) replaces the `-1` sentinel.
- `normalize` defaults to `False` everywhere, including server-level steering.

<!-- TODO: per-algorithm pages (file formats, payload shapes) — currently only the
README's "Adding a New Algorithm" snippet covers this. -->
