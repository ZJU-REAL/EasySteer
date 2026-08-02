# Migration plan: EasySteer → vLLM 0.26.0

*Drafted 2026-08-01. Base: `EasySteer-vllm-v1` @ `21d7a9f` (vLLM v0.17.1, wheel commit `95c0f92`). Target: upstream vLLM v0.26.0 (released 2026-07-25).*

## Status (2026-08-02): core migration validated ✅

- Branch `migrate-v0.26.0` pushed to `git@github.com:xuhaolei/EasySteer-vllm-v1.git`
  (`44ae813` port + `0b9e559` V1-runner fix).
- README sentiment example passes on vLLM 0.26.0 (env `easysteer-vllm026`,
  Python 3.12, torch 2.11/cu130, zju-53 GPU 0): unsteered output is
  byte-identical to the 0.17.1 baseline; steered output is correctly upbeat.
- Confirmed risk #1 in practice: Qwen2.5 defaulted to **Model Runner V2**,
  which silently skipped all steering hooks. Fixed by forcing the V1 runner
  whenever `steer_vector_config` is set (`vllm/config/vllm.py`).
- New install requirements vs 0.17.1: `pip install gguf` (0.26 dropped it),
  and steering requires `enable_prefix_caching=False` (fork-side validation
  added after the README example was written — README needs updating).
- **Model Runner V2 eager-mode steering: done and validated** (`ca47338`).
  New `vllm/v1/worker/gpu/steer_vector_utils.py` + V2 runner wiring; steered
  output on V2 is byte-identical to the V1 run. Runner policy: steering +
  `enforce_eager=True` → default selection (V2 where supported); steering +
  CUDA graphs/compiled execution → V1 with a warning.
- **Eager-mode refactor Phase A done** (`3d73f41`): decoder-layer steering
  now uses forward hooks instead of module replacement — state-dict keys,
  module names and classes untouched (FSDP/VERL-safe); no more nested
  double-wrapping with the capture wrapper. moe_layer still uses
  replacement (future: FusedMoE pre-hook). Validated byte-identical.
- **Phase B done** (`0e15762`): structural module discovery fallback
  (AttentionLayerBase / MoERunnerInterface anchors; class-name lists remain
  as override) — unlisted model families steer with zero config; dead
  LoRA-mimicry APIs removed; single LRU. Byte-identical goldens.
- **Schema consolidation done** (`d3b77a3`): canonical steering-parameter
  field registry in request.py with import-time drift guards; ~40
  hand-written field copies replaced by registry helpers; fixed latent
  multi-vector drift bug (dropped generate_first_k/after_k).
- **Mask-algebra rewrite done** (`bab5e8c`): position collector 695→350
  lines under an explicit algebra; legacy quirks documented; pre-refactor
  code kept as fuzz oracle (scripts/test_mask_equivalence.py, 5000 cases,
  0 mismatches); golden + trace oracle PASS.
  Remaining refactor backlog (agreed order): legacy-activation retirement
  (after CUDA-graph design) → FusedMoE pre-hook port → capture rework
  (see Future requirements).
- **Phase C done** (`e80aefa`): per-request steering configs under
  continuous batching. VectorStore (preload API, LRU by vector file,
  `max_steer_vectors` default 8), fingerprint→slot routing with per-slot
  trigger controllers, steering trace (`VLLM_STEER_TRACE_DIR`).
  Validated: golden byte-identical; trigger-position trace oracle exact
  (positions/phases/layers, unsteered request untouched); 51-scale batch
  8.8× faster than sequential with scale-0 byte-equal to unsteered.
  Known limits: multi-vector + moe_router configs fall back to legacy
  global activation; text-level batched-vs-sequential equality is bounded
  by vLLM's residual batch-shape nondeterminism (even with
  VLLM_BATCH_INVARIANT=1); `generate_first_k=k` effectively steers k−1
  decode steps (V1-parity boundary quirk, documented in the test).
## CUDA-graph steering design (data-driven, 2026-08-02)

Principle: **captured graphs contain no steering configuration — only
kernels that read buffers**; all trigger/routing decisions are computed
host-side before the forward pass (enabled by Phase C slot routing).
Replaces the legacy bake-in (`steer_allow_cuda_graphs`), which required
recapture per config and blocked per-request steering.

- **Mechanism**: single opaque custom op `vllm::steer_apply(hidden,
  layer_name)` (steer_vectors/ops.py, mutating, fake-impl no-op). The
  forward hook body stays dynamo-traceable (format unpack + add + op
  call); dynamo inlines the hook so the op lands in the FX graph; all
  steering logic runs inside the op impl via a layer-name→controller
  registry.
- **Tier 2 done** (`2742cb3`), validated on zju-53 GPU 0: eager golden
  byte-identical after the custom-op refactor; under compiled+piecewise
  execution steering fires (hook traces into the FX graph), scale-0 is
  byte-equal to unsteered, replay is deterministic, and the
  trigger-position trace oracle passes exactly; mask fuzz 0/5000.
  Compiled outputs differ textually from eager goldens (expected
  compiled-kernel numeric drift). Fixed: trigger masks are now sized
  from current_tokens, not the padding-bucket hidden states.
- **Tier 2 mechanism**: with steering + compiled
  execution, config validation appends `vllm::steer_apply` to
  `splitting_ops` and clamps `cudagraph_mode` to PIECEWISE — the graph
  splits at every steered layer and steering runs eagerly between
  CUDA-graph segments with full ForwardContext access. All algorithms,
  per-request routing and the steering trace work unchanged. V2 runner
  only; V1-runner configs fall back to enforce_eager.
- **Tier 1 (planned)**: full-graph fast path for graph-safe (additive)
  algorithms via persistent buffers — per-layer vector table
  `V[max_slots+1, hidden]` (row 0 = zeros), per-step `slot_tok` +
  effective-scale buffer `S[num_layers, max_tokens]` folding triggers/
  target-layers/scale; in-graph kernel `hidden += S[l]·V[slot_tok]`.
  Gated by per-algorithm `graph_safe` flag; hidden-state-conditional
  steering expressible branch-free (soft gates) also qualifies.
- **Legacy-activation retirement done** (`4b49a74`), validated on
  zju-53 GPU 0: eager golden byte-identical; trigger trace oracle PASS
  (eager + compiled); multi-vector routed PASS in eager and piecewise
  (single-sub-vector case byte-equal to the single-vector path); server
  default slot byte-matches the golden per-request equivalent.
  All steering is slot-routed; V2 is the only steering runner
  (V1 hot-swap wiring removed, `VLLM_USE_V2_MODEL_RUNNER=0` + steering
  raises). multi_vector ports to per-slot sub-algorithm lists (payloads
  deduped via VectorStore); moe_router keeps wrapper-level single-active
  semantics (newest distinct config overwrites, with warning; restored
  on release); server-level steering occupies a default routing slot
  that requests without their own config are routed to (the
  add/remove_steer_vector RPCs now set/clear it). Deleted: adapter
  activate/deactivate machinery, LRU managers/caches, `_load_adapter`,
  V1 per-step hot-swap; `steer_allow_cuda_graphs` is a deprecated
  no-op.
- **Tier 1 done** (`4b49a74`, `steer_graph_mode=full`), validated on
  zju-53 GPU 0 in eager and compiled modes: cudagraph_mode stays
  FULL_AND_PIECEWISE, steering fires under FULL graph replay, scale-0
  is byte-identical to unsteered, replay deterministic, routing
  isolation holds in mixed batches, non-graph-safe configs (loreft)
  rejected at the frontend. Mechanism: per-layer persistent
  vector tables `V[rows+1, hidden]` (row 0 zero, pre-scaled payloads) +
  per-layer trigger-mask buffers + shared token→row buffer; the hook
  computes `hidden += mask * V[row_tok]` — pure tensor math that
  captures into full CUDA graphs and compiled code (no graph splits).
  Triggers/routing are computed host-side per step
  (fill_graph_steer_buffers, same collector as eager). Graph-safe
  configs only (direct, no normalize, single-vector) — rejected at the
  frontend otherwise. Steering trace is unavailable in full mode.

- **Cleanup package done** (validated on zju-53/zju-48):
  - `2251496` — slots are ordered intervention lists on the layer
    controller (single-vector = list of one, conflict resolution is a
    slot property); MultiVectorAlgorithm deleted; template lost its
    per-slot machinery (context_info form kept for the MoE wrapper).
  - `8f99ea1` — VectorStore keyed by (path, algorithm, mtime_ns, size)
    + `reload()`; file version also in config_fingerprint, so vectors
    regenerated at the same path load fresh (dynamic-vector workflows).
  - `abc85be` — `--steer-require-preload` (opt-in frontend rejection of
    unpreloaded vectors; lazy load stays the default) + GET/POST
    `/v1/steering/vectors` admin endpoint.
- Deferred by user decision: `-1` trigger-token API cleanup — to be
  folded into a future unified redesign of the user-facing API across
  eager and CUDA-graph modes (incl. whether `[-1]`-bypasses-exclusions
  survives).

- **MoE gate-hook port done** (`384d5b3`): MoE router-logits steering
  is a forward hook on each block's gate/router submodule (logits
  steered in place before top-k) via the `vllm::steer_moe_gate`
  splitting op — architecture-agnostic through structural FusedMoE
  discovery; moe_layers.py (740 lines of per-architecture forwards)
  deleted. Dense regression green; MoE runtime validation done (see
  SteerMoE entry below).
- **Capture rework done** (`6d3dc11`): hook-based CaptureSession with
  `hidden_states` + `router_logits` streams; per-stream layer subset,
  dtype, per-sample position reduction (all|last|mean) and per-layer
  token budget; chunk-append storage (no batch heuristics); V2-runner
  support; legacy RPC names kept as shims (same wire format), so the
  easysteer client works unchanged. Router-logit capture is now
  architecture-agnostic (shared gate discovery) and records
  post-steering logits. 11-check capture test PASS + full steering
  regression green on zju-53.
- **MoE runtime validation done via SteerMoE replication** (`6394556`,
  test_steermoe_olmoe.py, OLMoE-1B-7B-0125-Instruct on zju-53, all 8
  checks PASS): router_logits capture (16 layers x 64 experts, exact
  row counts), risk-difference expert detection from one contrastive
  pair (paper arXiv:2509.09660, reference repo
  adobe-research/SteerMoE), new paper-exact `steermoe` moe_router mode
  (log-softmax, activated -> max+eps / deactivated -> min-eps),
  deterministic mechanism check (post-steering captured logits exclude
  all 40 deactivated experts from every token's top-8), behavior
  demos. Also fixed: request `moe_mode` no longer force-overrides
  per-layer JSON modes (defaults None now); fused-gate guard warns
  loudly for shared-expert models whose MoE runner bypasses the gate
  module (`_fse_fuse_gate`) — gate hooks (steering AND capture) cannot
  work there (e.g. Qwen1.5/2-MoE-style blocks).
  `replications/steermoe/` entry done (`c6bdb25` in the EasySteer root
  repo, committed locally — no push rights to ZJU-REAL origin):
  README + executed notebooks (detection via router_logits capture,
  steering via steermoe mode). Tuning sweeps established: 3
  contrastive pairs + 100 deactivated experts flips OLMoE counting
  from digits to words (0 post-steering top-8 leaks); activation-mode
  steering degrades output (paper agrees: deactivation preferable);
  faithfulness pkl orientation is positive-Δ = faithful-linked, and
  OLMoE's baseline is near ceiling on short counterfactual QA, so the
  notebook demos the visible unfaithful direction with a ceiling note
  (the S6 'flip' in the first validation run was this ceiling effect,
  not a sign error).

- **Slot-routed MoE steering done** (`b7d9956`): MoE gate steering
  routes per-request through the shared slot machinery — new
  `SlotRoutedSteerController` base in layers.py holds the slot engine;
  decoder and gate controllers are thin subclasses (router-logit rows
  share the token layout of hidden states, so the routing loop applies
  unchanged; MoE steering gains trace support). Single-active
  semantics and the last legacy application path
  (`apply_intervention`/`context_info`/`layer_apply_kwargs`) deleted;
  distinct MoE configs now batch together, each steering only its own
  requests' tokens. Missing `list_steer_vectors` worker RPC delegate
  added. Validated on zju-53: test_moe_slot_routing.py R1–R4
  (deterministic per-row attribution via the steermoe bottom-set
  signature — non-strict dominance, since bf16 rounds the eps margin
  away) + full regression (steermoe olmoe, dense golden
  byte-identical, multi-vector, capture, Tier-1 full-graph, compiled
  trigger oracle, require-preload). Gotchas: async scheduling staggers
  prefill admission, so multi-request byte-compares across generate
  calls are flaky — test_multi_vector_routed.py now pins
  `async_scheduling=False`; killing the reservation server's launcher
  pid can orphan its EngineCore still holding GPU memory — lab
  drivers now wait for GPU-0 memory release and reap stragglers.

- **MoE mode rename + extended testing done** (`1d9263b` vllm-steer,
  `b74ccf1` EasySteer): canonical moe_router modes are `activate` /
  `deactivate` (paper terminology, log-softmax ±eps mechanism);
  `boost`/`soft_hard` → activate, `suppress` → deactivate, `steermoe`
  accepted — aliases byte-identical (validated); on overlap
  deactivation wins. Validation scripts moved to `EasySteer/tests/`
  (env-parametrized paths + `STEER_TEST_TP`); replication notebooks
  re-executed with canonical modes. New coverage all green:
  alias equivalence, mixed-direction layers, no-file request path,
  trigger-conditioned gate steering, compiled-piecewise MoE
  (trace-verified), Qwen3-MoE smoke (Qwen3-30B-A3B non-2507 — the
  paper's exact model, enable_thinking=False; internal-router gate
  path), and **TP=2 on zju-57 GPUs 2/3** (multi-vector, capture, MoE
  modes, per-request MoE routing, Qwen3-30B — all PASS). zju-56 (4x
  RTX 3090) cannot run the cu130 env (driver 570 < 580). Reservation
  server on zju-53 currently STOPPED at user request. Remaining
  parallelism gap: expert parallelism (EP) untested.

- **Codebase cleanup wave done** (`26d5888` vllm-steer, `10a5492`
  EasySteer tests): dead wrapper mode + slot-base offset + id counter
  removed; AlgorithmTemplate single-payload (`set_payload`);
  hidden_states request/deserializer dedup (moe twins deleted, names
  aliased for the legacy client); discovery structural-first with
  class-name lists as fallback (models.py + capture.py, registry
  indirection gone); full upstream style pass (ruff format, modern
  typing, init_logger, lazy logging, 88-char, banners/stale docstrings
  pruned; E/F/UP/G004/E501 clean). Net -230 lines, validated: CPU
  units + eight GPU suites green; three apparent failures on GPU 5
  were cross-GPU kernel numerics (goldens are GPU-model-specific) and
  re-passed byte-identical/green on GPU 0. Remaining style-adjacent
  backlog: dual pydantic/msgspec schema (parked with the API
  redesign), soft_topk deprecation question.

- **Naming/organization refactor done** (`543da11` vllm-steer,
  `9b678a4` EasySteer tests): new `steer_vectors/discovery.py` (model
  introspection shared by steering and capture — structural finders +
  class-name lists absorbing config.py, decoder-output
  split/reconstruct made public, gate-logits extraction, forward-pass
  sample helpers from algorithms/utils.py; capture no longer imports
  steering privates); controller terminology (wrapper_type →
  controller_type, _adapter_manager → _controller_manager, empty
  BaseLayerWithSteerVector shim and factory.py stub class deleted);
  vocabulary renames (parameter_control.py → triggers.py with
  TriggerController, SteerVectorModel → LoadedSteerVector,
  SteerVectorModelManager → SteerControllerManager, create_sv_manager
  → create_steer_controller_manager); package rename
  `vllm/hidden_states` → `vllm/capture` (session.py/serde.py/
  request.py) with `vllm.hidden_states` kept as a re-export alias and
  legacy RPC names unchanged; capture docstring documents relation to
  upstream's `extract_hidden_states` speculative method (identical
  hidden+residual composition). Validated: CPU units + seven GPU
  suites PASS on zju-53 GPU 5; golden byte-identical +
  server_default_slot PASS on GPU 0.

- **Algorithms package cleanup done** (`25db19d` vllm-steer, `8fe5d81`
  EasySteer tests): shared `algorithms/loading.py` (GGUF reader was
  copy-pasted 4x with a dead arch-check block; ReFT checkpoint
  discovery duplicated 2x); unified `load_from_path(path, device, *,
  config, target_layers, **kwargs)` signature + documented
  `layer_payloads` return contract; `AlgorithmTemplate.params` →
  `.triggers` (ended the params-means-two-things collision); shared
  `_renormalize` (float32 + eps) replacing three divergent normalize
  implementations (fixes fp16-overflow exposure in
  erase/replace/concept_replace; byte-identical on the direct golden).
  Explicit-failure policy applied: no 48/32-layer fallbacks in
  linear/lm_steer, moe_router raises on invalid layer ids / unknown
  modes / missing expert ids (validated at load). Broad
  `except Exception` re-wraps removed (linear, lm_steer, moe_router,
  from_local_checkpoint); discovery forward-context getters catch only
  AssertionError. Net −470 lines. New CPU unit
  `test_algorithm_loaders.py` (21 checks). Full battery green (GPU 5
  suites, golden byte-identical + server-slot on GPU 0).
  PENDING per user: a dedicated repo-wide review of implicit-default
  patterns (prefer explicit failure) beyond the algorithms package.

- **Explicit-failure + phase/chunked-prefill fixes done** (`309e1a2`
  vllm-steer, `6c2b4a7` EasySteer): triggerless steering configs,
  moe_router-no-file-without-target_layers, and empty layer_payloads
  now raise at construction/load instead of silently steering nothing;
  `steer_vector_dtype="auto"` resolves to the model dtype in VllmConfig
  post-init (was silently fp16 — goldens re-recorded on GPU 0 with
  bf16 vectors, fp16 golden kept as golden.txt.fp16-bak; baseline was
  byte-identical, exactly one steered section drifted).
  Prefill/decode classification now uses scheduler ground truth
  (`num_output_tokens > 0` in extract_samples_info; is_prefilling_np
  in fill_graph_steer_buffers) — fixes 1-token prompts and 1-token
  chunked-prefill tails. **Chunked prefill is now supported with
  steering** (legacy ban lifted): new ForwardContext
  num_prompt_tokens_cpu makes negative prefill positions
  prompt-relative (previously they re-fired per chunk); prefix-caching
  rejection unchanged (fundamental). New trace-oracle suite
  test_phase_chunked_prefill.py (P0–P6) PASS; full battery green.
  Migration note for old app code: hf-space/app.py builds a
  triggerless scale-0 baseline request that will be rejected by this
  fork — add prefill/generate_trigger_tokens=[-1] when migrating it.

- **Prefix caching supported with steering; capture guarded**
  (`cdbe8d0` vllm-steer, `a4aba41` EasySteer tests): block hashes key
  steered requests by `config_fingerprint` (replacing the
  name-label key that caused 9b999cb); prompt-length-sensitive configs
  (negative positions, first_k/after_k) additionally key on prompt
  length; blocks past the prompt boundary carry the producer's prompt
  length (phase safety for continuation prompts). Server-level mode
  uses upstream salt+reset: engine-core salts every hash with the
  startup server fingerprint (`steer_vectors/cache_salt.py`),
  POST /v1/steering scale updates call `reset_prefix_cache`, fresh
  runtime installs on caching engines rejected. Capture: VERIFIED that
  prefix-cache hits silently skip capture (11/43 rows on warm cache) —
  enabling capture on a caching engine now raises; `last` reduction is
  chunk-aware; `mean` warns per-chunk under chunking; V2 capture-only
  path passes full geometry. Validated: S/R/K mechanism suites
  (num_cached_tokens oracle) + full battery green, golden
  byte-identical on GPU 2. Remaining: length-sensitive configs share
  cache only at equal prompt lengths; `mean` under chunking stays
  per-chunk; V1-runner capture keeps the legacy phase heuristic;
  in-flight requests during a scale update keep old-scale KV
  (pre-existing runtime-mutation semantics); live HTTP test of
  /v1/steering + reset still on the backlog.

- **Steering API v2 (user-facing redesign; v1 deprecated)** — design
  in `STEERING_API_V2.md`. Three concepts (`SteeringSpec`/`VectorSpec`/
  `ApplySpec`, pydantic, exported from `vllm.steer_vectors`; single
  authoring schema for offline + HTTP) replace the v1 pile: phases
  enum replaces the `-1` sentinel; exclusions always compose (v1
  bypass gone); `generation_window=(start,stop)` half-open over decode
  steps replaces first_k/after_k and fixes the k−1 quirk; params dict
  folds the moe_* fields; no `"path|algo"`, no user-visible name/id;
  normalize defaults False everywhere. Implementation: specs translate
  at admission (`api.py to_engine_request`); the where-clause travels
  as one canonical `apply_spec` dict on the internal struct
  (registered in STEER_TRIGGER_FIELDS → fingerprint/slot config/
  configure_from_dict propagate automatically; length-sensitivity
  handled) and executes natively in a dedicated v2 collector shared by
  eager/piecewise/full-graph. Entry points: `llm.generate(...,
  steering=)`, HTTP `"steering"` field, engine default
  `--steering-config` (SteeringSpec JSON, inline or file; normalized
  at config validation; v2 branch of build_server_request feeds the
  salt + worker install), POST /v1/steering `{"spec": {...}}` full
  replacement (install-first, then commit + reset_prefix_cache).
  Deprecated with warnings: `steer_vector_request` (offline + HTTP),
  `--steer-vector-path` flag family, scale-only POST. Tests:
  test_api_v2_unit.py (38 CPU checks), test_api_v2.py (W1–W10 trace
  oracle), test_api_v2_server.py (X0–X4), prefix-cache S9/S10.
  Deletion of v1 (fields, v1 collector, fuzz oracle, Param twins)
  waits for easysteer pkg/notebook/hf-space migration.

## Future requirements (recorded, not current priority)

- **Dedicated documentation site**: rather than extending the current
  README-based docs, we may build a dedicated project page with detailed
  documentation similar in scope and structure to the vLLM docs site
  (user guide / API reference / algorithm catalog / replication index).
  The v2 API design doc (STEERING_API_V2.md) and the standardized
  notebooks are written to be liftable into such a site.
- **Frontend + hf-space adaptation to v2** (deferred; not urgent): both
  still use the v1 request JSON (`frontend/core/steer_request_builder.py`,
  `frontend/chat_api.py`, `frontend/inference_api.py`, `hf-space/app.py` —
  note its triggerless scale-0 baseline request also needs an `apply`
  clause under v2). The frontend is additionally slated for a visual
  overhaul later; adapt API + looks together.

- Remaining (not yet validated): Tier-1 full-graph steering kernel
  (planned; legacy `steer_allow_cuda_graphs` bake-in slated for
  retirement), OpenAI server-level steering endpoints, MoE
  capture/steering, VLMs,
  `easysteer` pkg + pyreft under transformers v5. Hidden-states capture
  (CaptureModelRunnerMixin) is still V1-only — set
  `VLLM_USE_V2_MODEL_RUNNER=0` when using the capture APIs on
  default-V2 architectures.

## 1. Current state

- All vLLM integration lives in the `vllm-steer/` submodule fork; the `easysteer` package only imports `vllm.hidden_states` and migrates for free.
- The fork is almost purely additive: **+10,140 / −9 lines over 70 files, all Python** (no csrc/cmake changes → precompiled wheels usable).
  - Self-contained new packages (~80% of diff, copy over nearly unchanged): `vllm/steer_vectors/`, `vllm/hidden_states/`, `vllm/v1/worker/{capture,steer_vector}_model_runner_mixin.py`, `vllm/config/steer_vector.py`, tests, scripts.
  - ~20 surgical touch points (the real migration work): `forward_context.py` (4 new `ForwardContext` fields), `v1/worker/gpu_model_runner.py`, `gpu_input_batch.py`, `gpu_worker.py`, v1 scheduler/engine-core plumbing, `engine/arg_utils.py`, `entrypoints/llm.py`, OpenAI server files (server-level steering).

## 2. Relevant upstream changes 0.18 → 0.26

| Release | Change | Impact |
|---|---|---|
| 0.22–0.23 | **Model Runner V2** default for Qwen3/Llama/Mistral dense | **Highest risk** — fork hooks live in V1 `gpu_model_runner`; steering silently bypassed on V2 paths |
| 0.19 | KV cache refactor (list → element); virtual engine deprecated | Rework fork's `kv_cache_utils`/block-pool touches |
| 0.20 | Transformers v5 baseline; PyTorch 2.11 + CUDA 13.0 default wheels | Driver ≥ 580 needed for default wheels; check `easysteer.reft`/pyreft vs transformers v5 |
| 0.21 | C++20 for source builds | Moot if precompiled wheels used |
| 0.23 | Pluggable `KVCacheSpec`; stricter config validation | Touch-point friction |
| 0.24 | vLLM no longer sets `CUDA_VISIBLE_DEVICES` (`device_ids` arg) | Update experiment launch scripts |
| 0.25 | Legacy `api_server.py` deprecated; PagedAttention (V0) deleted | Move server-level steering endpoints to new entrypoint |

## 3. Strategy

Branch from the **v0.26.0 tag** and re-apply the fork as a curated patch series (`git diff v0.17.1 <fork-head>`, applied with `git apply --3way`), rather than rebasing fork commits through nine releases. The additive packages land cleanly; hand-port the ~20 touch-point files against the refactored internals.

## 4. Phases

- **Phase 0 — Baseline**: reproduce current behavior on the 0.17.1 stack (existing `easysteer-0.17.1` env on zju-48); record golden outputs (fixed seed, greedy) for representative steering configs; run fork test suite.
- **Phase 1 — Mechanical port**: new branch from v0.26.0; copy self-contained packages; 3-way-apply touch-point patches; engine must boot with steering disabled and match vanilla 0.26.0 output.
- **Phase 2 — Hot spots** (risk order):
  1. Model runner: rewire mixins; interim policy = force V1 runner path when `steer_vector_config` is set, port to Runner V2 later.
  2. `ForwardContext` fields re-derived from 0.26 batch structures; re-verify prefix-caching / continuous-batching trigger semantics.
  3. Scheduler/engine plumbing for per-request steer/capture state.
  4. OpenAI server steering on non-deprecated entrypoint.
  5. CUDA-graph capture path re-validation.
- **Phase 3 — Validate**: fork tests + golden comparison (exact match for deterministic configs; else tight logprob tolerance); `bench_eager_vs_cudagraphs.py`; one model per family (dense / MoE / VLM). Minimal gate: README sentiment-steering example produces correct steered output.
- **Phase 4 — Release chores**: transformers v5 deps, README install (`VLLM_PRECOMPILED_WHEEL_COMMIT` → v0.26.0 commit), Docker image, submodule pointer.

## 5. Lab environment notes

- Fork is pure Python → `VLLM_USE_PRECOMPILED=1 pip install -e .` works; install on **zju-48** (home hosted there).
- Driver gap for CUDA 13 default wheels (need ≥ 580): zju-48 = 580.105 ✓; zju-46 = 575, zju-54 = 570, zju-12 = 560 ✗. Options: validate on zju-48; use cu128 wheel variants for the wider fleet; or request driver upgrades.

## 6. Risks

1. Model Runner V2 silently bypassing hooks → force-V1 interim policy + assertion that wrapping occurred.
2. Trigger-position semantics under 0.26 prefix caching / chunked prefill → golden suite.
3. `torch.compile` / CUDA-graph interaction with model wrapping → test both modes early.
4. Transformers v5 breakage in bundled `pyreft`.
