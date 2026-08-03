# Capture Module Redesign Proposal

Status: draft for discussion (2026-08-03). Companion to `MIGRATION_PLAN_vllm-0.26.0.md`.
Scope: joint redesign of engine-side capture (`vllm-steer/vllm/capture/`) and the
consumer side (`easysteer/hidden_states/`, `easysteer/steer/` extractors), based on a
survey of all 41 notebooks, the frontend extraction pipeline, and the test suite.

---

## 1. Evidence: how capture is actually used

- 13 notebooks touch capture. 11 use `get_all_hidden_states_generate`; of those, **10
  call it with defaults** — full-sequence, all-layer capture — and then keep roughly
  one row per sample (`token_pos=-1` in the extractor). The one exception
  (`experiment/math`) is the only user of source-side `token_ids` selection.
- The client wrapper does not expose `layers`, `dtype`, `max_tokens`, or the
  `positions="last"/"mean"` reductions at all, even though the engine supports them.
  The embed-path API (`get_all_hidden_states`) has zero notebook users and is already
  marked for deprecation.
- Usage archetypes:
  - **A. Paired pos/neg, last-token → gguf** (7 notebooks): diffmean / PCA / probe on
    one row per sample. Needs `positions="last"` (refusal_direction: last 4).
  - **B. Last-k mean** (cast): client-side `[-4:].mean(0)` plus a fabricated nested
    list to satisfy the extractor signature.
  - **C. Category-classified interior positions in generated transcripts** (seal,
    math): the demanding case; math needed chunk loops, running sums, and a
    position-rank realignment because selected rows come back **unlabeled**.
  - **D. One layer, one position, mean+covariance** (sake): captures ~33 layers ×
    all tokens to use exactly one row per sample.
  - **E. MoE router-logit top-k counting** (steermoe ×2 + 3 tests): bypasses
    easysteer entirely, hand-rolls the stream RPCs, keeps only a (layers × experts)
    count matrix.
- Reducibility of the algorithms:
  - Fully reducible to running statistics: diffmean (sum+count), PCA
    standard/center/diff (second moments; center/diff require *pairing* to be
    preserved), SAKE (mean+cov), steermoe detection (count matrix), SEAL/math
    category means.
  - Genuinely need per-**sample** rows (not per-token): LAT (per-row normalization +
    shuffle before PCA), linear probe (iterative LR fit).
  - Nothing in the repo needs full per-token capture except transient inspection.

## 2. Diagnosed design problems

1. **Selection lives at the wrong layer.** The extractors' `token_pos` does row
   selection client-side after the full transfer; the engine-side selection that
   would make it free is unreachable through the wrapper. >99% of transferred data
   is discarded in the dominant archetype.
2. **Rows are unlabeled.** Fetch returns bare `[rows, dim]` per layer; sample
   splitting is re-derived client-side from token counts, three times, in three
   places (`capture.py` heuristic that silently fudges the last sample,
   `capture_generate._selected_sample_lengths` mirroring engine semantics, the math
   notebook's rank lookup + mismatch counters). This is the root cause of the whole
   realignment/workaround class and violates the fail-explicit policy.
   **Confirmed during P1 implementation (2026-08-03): this is not just a
   workaround burden but a live correctness bug.** The store appends rows
   step-by-step, so decode rows of concurrently-generating requests interleave;
   the client splits contiguously by per-sample expected lengths, so under
   continuous batching rows are attributed to the wrong samples — and the
   expected-length construction makes the per-sample count check pass by
   definition, hiding it. Any multi-request generation capture (SEAL math run
   included) was affected. Labeled rows are the fix; affected artifacts must be
   regenerated.
3. **Two parallel APIs have drifted.** easysteer speaks only the legacy per-feature
   RPCs; tests and steermoe use the stream API. Inconsistent `split_by_samples`
   defaults (hidden states: True, router logits: False); hidden states return
   layer-*lists* indexed positionally (wrong under layer subsets) while router
   logits return dicts keyed by true layer id.
4. **The extractor input contract forces materialization.** `[sample][layer][token]`
   nested Python lists means everything must exist in client RAM at once; cast
   fabricates fake nesting to comply, seal/math bypass extractors entirely and
   re-implement averaging by hand.
5. **Storage is unbounded engine-side CPU RAM** with no spill, and fetch is a
   monolithic rank-0 msgpack RPC (per-layer fetch added recently mitigates, but the
   model is still "hold everything until the client pulls").
6. **Selection semantics are duplicated.** Capture's `positions`/`token_ids` is an
   ad-hoc subset of steering's `ApplySpec` (phases, tokens, positions with
   negative-from-prompt-end, exclusions, generation window), implemented with the
   same trigger helpers but a different surface, different validation, different
   docs.

## 3. Proposed design

### 3.1 One selection language: `SelectSpec` shared with steering

Factor the where-clause out of steering's `ApplySpec` into a `SelectSpec` (phases,
tokens, positions, exclude_tokens, exclude_positions, generation_window) used
verbatim by both steering (`ApplySpec = SelectSpec + nothing`) and capture. One
implementation (triggers.py), one validation, one semantics doc. Capture drops its
ad-hoc `positions="all"|"last"|"mean" | list` + `token_ids` kwargs in favor of
`select: SelectSpec` + `reduce:` (below).

### 3.2 `CaptureSpec` mirroring `SteeringSpec`

```python
class StreamSpec(BaseModel):
    what: Literal["hidden_states", "router_logits"]   # extensible
    layers: list[int] | None = None
    select: SelectSpec | None = None                  # None = all rows
    reduce: Literal["none", "last", "mean"] = "none"  # within-sample only
    dtype: str | None = None
    budget_rows: int | None = None

class CaptureSpec(BaseModel):
    streams: list[StreamSpec]
```

Attached **per request** (`llm.generate(..., capture=spec)`), exactly like
`steering=`; an engine-wide default remains possible but is no longer the only mode.
Per-request attachment gives per-request position lists for free, scopes lifecycle
(no global enable/disable leakage between users), and makes capture composable with
steering in one call (the steermoe verification pattern).

### 3.3 Labeled rows

Every captured row carries metadata columns: `(request_id/sample_index,
absolute_position, token_id)` as int32 arrays alongside the float rows. Fetch
returns per layer `{rows, meta}`; the client never re-derives alignment. Cost is
~8 bytes/row against 2–8 KB/row of activations. All three client-side realignment
mechanisms are deleted; mismatch is impossible by construction rather than counted
defensively.

### 3.4 The extraction / post-processing boundary (decided)

Decision (2026-08-03): extraction and downstream post-processing keep a hard
boundary — **no cross-sample computation in the capture layer**. The engine may
select rows and reduce *within one sample's geometry* (`layers`, `select`,
`last`/`mean`, dtype) because that determines what leaves the GPU; it never
computes corpus-level statistics, never sees group/category/pair semantics, and
never learns what a "positive" sample is.

Cross-sample aggregation lives entirely in easysteer as **streaming accumulators**
(`DiffMeanAccumulator`, moments-based PCA fit, steermoe's layer×expert count
matrix) that consume streamed per-request results incrementally — client memory
stays O(1); transport is O(samples), which after per-sample reduction is small
(worst surveyed case ≈ 1 GB total, streamed). `from_moments` constructors are
easysteer-internal plumbing between its accumulators and its solvers.

Rationale: the engine spec stays free of algorithm churn (new extractors are
easysteer PRs, never fork PRs), no statistics-equivalence test burden on the
engine, no pairing protocol leaking into the wire format, and non-reducible
algorithms (LAT, probes) need the raw per-sample path anyway.

### 3.5 Transport, storage, retrieval

- Keep the raw-dtype wire encoding and per-layer fetch (already done).
- Fetch model becomes **per-request-id or per-layer streaming**: the client pulls
  results incrementally as requests finish (aligns with vLLM's own output
  streaming), instead of one accumulate-then-drain store.
- Budgeted stores fail explicitly when exceeded (keep `budget_rows` + drop
  accounting), and an optional safetensors spill directory covers
  bigger-than-RAM raw captures (P4, only if a real workload needs it).
- Per-sample reduction keeps wire volume small enough that shm/CUDA-IPC
  transport is unnecessary; defer it (P4). End-state retrieval shape: a
  `CaptureRef` on each `RequestOutput` — small payloads inline, large payloads
  as a resolvable handle (shm segment / spilled file / in-process tensor).

### 3.6 easysteer client rework (joint with extractors)

- One entry point: `CaptureResult = hs.capture(llm, prompts, spec, sampling=...)`
  (or `llm.generate(..., capture=spec)` passthrough). `CaptureResult` is indexable
  by sample and layer (dict keyed by **true layer id** everywhere), exposes
  `.rows(layer)` and `.meta(layer)`, and hides chunking/streaming entirely.
- Cross-sample aggregation (per section 3.4) happens here: streaming accumulators
  (`DiffMeanAccumulator`, moments-based PCA, expert count matrices) consume
  per-request results incrementally; `from_moments` connects them to the solvers.
- Extractors accept either per-layer `(n_samples, dim)` matrices or a
  `CaptureResult` directly; `token_pos` becomes a deprecated compatibility shim
  (selection belongs in `SelectSpec`). LAT/probe keep the matrix path.
- Unify the four capture classes into one; embed path removed (0 users); MoE router
  logits use the same result type with `what="router_logits"`; `analyze_expert_usage`
  reworked as a streaming accumulator (its current form has no callers).
- Legacy RPC shims and `get_all_hidden_states_generate` kept working for one
  release, then removed; the notebooks are the migration test corpus.

## 4. Phasing

- **P1 (highest value, no breaking changes):** labeled rows + expose
  `layers`/`select`/`reduce` through the client + `SelectSpec` factored out and
  shared with steering. Kills the realignment hacks and the 99%-discard transfers.
- **P2:** per-request `CaptureSpec`, `CaptureResult` client object, extractor input
  rework (matrices / CaptureResult), unified layer-id indexing and defaults.
- **P3:** streaming per-request retrieval (`CaptureRef` on `RequestOutput`,
  inline resolver) + easysteer streaming accumulators and `from_moments`
  extractor constructors.
- **P4 (on demand):** safetensors spill, shm transport, additional `what=` streams
  (mlp_out/attn_out), compiled-graph capture story (upstream
  `extract_hidden_states` KV-connector path for bulk offline extraction).

## 5. Known constraints carried forward

- Hook-based capture still requires `enforce_eager=True` and
  `enable_prefix_caching=False`; both remain admission-checked, fail-explicit.
- `center`/`diff` PCA needs pair identity; with aggregation client-side this is
  purely an easysteer bookkeeping concern (pair ids in the accumulator), never
  visible to the engine.
- TP>1: capture currently reads rank 0 only; labeled rows make multi-rank merging
  tractable later but it stays out of scope until a workload needs it.
