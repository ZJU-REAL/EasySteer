# Steering API v2 — design and migration

Status: **shipped; historical design record.** v2 is the only API
(since 2026-08-03). The v1 surface — trigger fields,
`steer_vector_request` (offline and HTTP), the pydantic Param twins,
the `--steer-vector-path` flag family and the v1 position collector —
has been deleted; notebooks, tests, docs and the docker smoke test all
use v2. (`frontend/` and `hf-space/` adaptation is still deferred; see
MIGRATION_PLAN.)

This file records *why* the API looks the way it does and how v1 was
retired. For the current, maintained documentation of the API itself,
see `vllm-steer/docs/features/steer_vectors.md` (user guide),
`vllm-steer/docs/design/steer_vectors.md` (architecture), and
`docs/user-guide/steering.md` (EasySteer docs site). Two things changed
after this record was written: the graph tiers were renamed
(`full` → `in_graph`, `piecewise` → `split`, chosen via
`steer_graph_mode` ∈ {`auto`, `in_graph`, `split`}), and admission
became declaration-first (`steer_algorithms` is mandatory when steering
is enabled; undeclared algorithms are rejected on every engine).

## Why

The v1 surface accreted concepts users must not need: two request schemas
(pydantic + msgspec) held together by a registry assert; single-vector fields
*and* `vector_configs` as two shapes of one thing; a separate server-level
surface with different defaults (`server_normalize=True` vs request `False`);
seven trigger fields with a `-1` sentinel that also silently bypasses
exclusions; a `first_k` window that steers k−1 decode steps; vestigial
identity fields (`steer_vector_name`, `steer_vector_int_id`); an algorithm
embedded in the path (`"path|algo"`); and four MoE request fields. Internally
everything already routes through fingerprinted config slots — server-level
steering is just a config on the default slot — so the API distinctions are
historical, not architectural.

## Concept model (3 concepts)

```python
from vllm.steer_vectors import SteeringSpec, VectorSpec, ApplySpec

spec = SteeringSpec(vectors=[
    VectorSpec(
        source="vectors/happy.gguf",   # path only; no "path|algo"
        algorithm="direct",
        scale=0.5,
        layers=[10],
        normalize=False,
        apply=ApplySpec(prompt="all", generation="all"),
    ),
])
```

1. **`ApplySpec`** — *where and when* a vector applies. Replaces the seven
   trigger fields and their sentinels.
2. **`VectorSpec`** — one vector: source, algorithm, scale, layers,
   normalize, algorithm-specific `params`, and its `apply` clause.
3. **`SteeringSpec`** — an ordered list of `VectorSpec` plus a conflict
   policy. The single-vector case is a one-entry list, not a second schema.

Two attachment scopes, same object:

- per request: `llm.generate(..., steering=spec)` / HTTP `"steering": {...}`
- engine default: `--steering-config spec.json` (or inline JSON), replaced at
  runtime via `POST /v1/steering {"spec": {...}}` (prefix cache is reset).

## ApplySpec semantics (locked)

- `phases: list["prompt" | "generation"]` — required, non-empty. Explicit
  phase selection replaces the `-1` sentinel. Phases are named for what the
  tokens *are* (prompt vs generated), not how they execute (prefill/decode):
  chunked prefill, prefix caching and CUDA graphs never change meaning.
- `tokens: list[int] | None` — token-id allowlist (real ids only, `>= 0`).
- `positions: list[int] | None` — absolute sequence positions; negatives are
  Python-style from the end of the *prompt* (`-1` = last prompt token),
  stable across prefill chunks.
- Within the selected phases: no `tokens`/`positions` → all tokens of those
  phases; otherwise the union of token matches and position matches.
- `exclude_tokens` / `exclude_positions` — **always subtract, in every
  case.** The v1 "`-1` trigger bypasses exclusions" behavior is gone.
- `generation_window: (start, stop)` — half-open interval over 0-based
  decode steps (the step processing generated token *j* is in the window iff
  `start <= j < stop`; `stop=None` = unbounded). `(0, k)` steers exactly the
  first k decode steps — the v1 `first_k` off-by-one is gone. Requires
  `"generation"` in `phases`. Replaces `generate_first_k_tokens` /
  `generate_after_k_tokens`.

## Other locked decisions

- **Identity**: no user-visible name/int-id. The config fingerprint (already
  the identity for slots and prefix-cache keys) is the identity. `VectorSpec`
  keeps an optional `name` for logs only.
- **Defaults**: `normalize=False` everywhere. The v1 server default of `True`
  survives only in the deprecated `--steer-*` flags.
- **Algorithm params**: `VectorSpec.params` dict, validated per algorithm
  (moe_router: `expert_ids`, `mode`, `lambda`, `topk`; other algorithms
  accept no params — unknown keys fail loudly). The four `moe_*` request
  fields fold in here.
- **Backend invariant**: the spec is backend-independent. Eager engines and
  both graph tiers (now named `split` and `in_graph`) accept the same spec;
  a spec an engine cannot run is rejected at admission with an explicit
  error. (Since the declaration-first rework, admissibility is derived from
  the engine's `steer_algorithms` declaration and the in-graph kernel
  families with their per-payload conditions — see the current docs; the
  original "direct, no normalize, single vector" full-graph restriction no
  longer applies.) `--steer-graph-mode` is an engine-level optimization
  setting, never something that changes how a request is written.
- **Server + per-request coexistence** stays disallowed for now (same as
  v1); the fingerprint + salt machinery would support relaxing it later.
- **Capture** is unaffected (already unified under `CaptureSession`).

## Implementation architecture

The engine wire format (`SteerVectorRequest`, msgspec) stays as an internal
struct. v2 specs translate at admission via
`vllm.steer_vectors.api.to_engine_request()`:

- The where-clause travels as one canonical `apply_spec` dict field on the
  internal struct (registered in `STEER_TRIGGER_FIELDS`, so the fingerprint,
  slot configuration and `configure_from_dict` propagate it automatically).
  `apply_spec` and legacy trigger fields are mutually exclusive on a struct.
- `TriggerController` executes `apply_spec` natively with a dedicated v2
  position collector (exact semantics above). The v1 collector and its
  legacy-equivalence fuzz test were deleted together with v1. (Trigger
  resolution has since been reworked into a single host-side numpy pass
  per scheduler step; see `vllm-steer/docs/design/steer_vectors.md`.)
- The in-graph (formerly "full-graph") fill path reuses the same collector,
  so v2 specs work under eager and both graph tiers with no extra code.
- Prefix caching: `apply_spec` participates in the fingerprint via the field
  registry; negative positions and `generation_window` mark the config
  prompt-length-sensitive, same rules as v1.

## Deprecation history

1. 2026-08-03 (`9ef3909`): v2 introduced; v1 deprecated with warnings.
2. 2026-08-03: notebooks, tests, docs, docker test migrated to v2.
3. 2026-08-03: v1 deleted — trigger fields, the v1 collector, the legacy
   fuzz oracle, the `"path|algo"` hack, the pydantic Param twins, and the
   `--steer-*` server flag family; `apply_spec` is the only where-clause
   and `--steering-config` the only engine-default mechanism.
