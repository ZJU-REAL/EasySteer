# EasySteer validation tests

End-to-end validation scripts for EasySteer's vLLM fork (`vllm-steer/`).
Each script boots its own engine, prints per-check `OK:`/`FAIL:` lines and
an `OVERALL: PASS|FAIL` verdict, and exits nonzero on failure. They are
standalone scripts (not pytest) because each needs a dedicated engine
configuration and a GPU.

## Configuration

| Env var | Meaning | Default |
| --- | --- | --- |
| `GPU_ID` | GPU to run on (`CUDA_VISIBLE_DEVICES`) | `0` |
| `STEER_TEST_MODEL` | dense test model (Qwen2.5-1.5B-Instruct) | lab path |
| `STEER_TEST_VECTOR` | steering vector (gguf) for the dense tests | `~/EasySteer/vectors/happy_diffmean.gguf` |
| `STEER_TEST_MOE_MODEL` | MoE test model (OLMoE-1B-7B-0125-Instruct) | `~/models/OLMoE-1B-7B-0125-Instruct` |
| `STEER_TEST_QWEN3` | Qwen3-MoE smoke-test model | lab path |
| `STEER_TEST_EAGER` | `1` eager / `0` compiled (where supported) | `1` |
| `STEERMOE_PKL` | SteerMoE released rankings pickle (optional) | cwd |

## Suites

Dense decoder-path steering (Qwen2.5-1.5B):
- `test_piecewise_steering.py` — Tier-2 steering under piecewise CUDA graphs
- `test_fullgraph_steering.py` — Tier-1 full-graph steering (`--steer-graph-mode=full`)
- `test_trigger_positions.py` — trigger/exclusion position oracle vs steering trace (takes `--model`/`--vector`)
- `test_multi_vector_routed.py` — slot-routed multi-vector configs
- `test_per_request_scale_sweep.py` — per-request isolation across a 51-scale batch
- `test_server_default_slot.py` — server-level (default-slot) steering
- `test_require_preload.py` — `--steer-require-preload` frontend enforcement
- `test_phase_chunked_prefill.py` — trace-oracle phase classification and
  chunked-prefill correctness (1-token prompts, 1-token tail chunks,
  negative trigger positions), request-validation explicit failures
- `test_steering_prefix_cache.py` — steering-aware prefix caching:
  fingerprint-keyed KV reuse (`num_cached_tokens` oracle), length-
  sensitive and phase-boundary keying, capture/fresh-install rejections
- `test_server_prefix_cache.py` — server-level steering with prefix
  caching (startup salt; scale update + `reset_prefix_cache`)
- `verify_steering_correctness.py` — model/hardware sanity harness (argparse)

MoE gate steering + router-logit capture (OLMoE-1B-7B):
- `test_moe_modes.py` — activate/deactivate semantics, deprecated aliases, mixed configs, no-file requests, trigger positions
- `test_moe_slot_routing.py` — per-request MoE routing in mixed batches
- `test_steermoe_olmoe.py` — SteerMoE (arXiv:2509.09660) end-to-end replication checks
- `test_moe_compiled.py` — MoE gate steering under piecewise CUDA graphs (trace-verified)
- `test_qwen3moe_smoke.py` — second architecture (internal-router gate path)

Capture streams (dense):
- `test_capture_hidden_states.py` — hook-based capture: layer subsets, reductions, budgets, dtypes, legacy RPC shims
- `test_capture_chunked.py` — capture coverage and chunk-aware `last`
  reduction under chunked prefill

CPU-only units:
- `test_mask_equivalence.py` — 5000-case trigger-mask fuzz vs the legacy collector (`_legacy_position_collector.py`)
- `test_store_reload_unit.py` — VectorStore file-version reload semantics
- `test_algorithm_loaders.py` — every algorithm's `load_from_path` against synthetic files, including the explicit-failure paths

Benchmarks:
- `bench_eager_vs_cudagraphs.py` — decode throughput, eager vs CUDA graphs

## Notes

- Steering supports prefix caching (block hashes are keyed by the
  steering config fingerprint; server-level mode salts every hash and
  scale updates reset the cache) and chunked prefill. Capture requires
  `enforce_eager=True` and `enable_prefix_caching=False` (cache-hit
  tokens are never recomputed, so they cannot be captured — enabling a
  capture stream on a prefix-caching engine raises).
- Byte-exact cross-run comparisons need identical batch geometry: use
  `ignore_eos=True` and pin `async_scheduling=False` (async admission makes
  prefill co-batching timing-dependent).
- Stored goldens (`golden.txt` comparisons in the sentiment and
  server-slot tests) are GPU-model-specific: kernel numerics differ
  across GPU models, so compare only on the model that recorded the
  golden. Behavior-level checks (does the output change/flip) can also
  be hardware-sensitive; mechanism-level checks (captured logits,
  steering trace) are not. golden.txt was re-recorded 2026-08-02 after
  `steer_vector_dtype="auto"` started resolving to the model dtype
  (bf16 vectors); the fp16-era golden is kept as
  `golden.txt.fp16-bak`.
- Steering requests must set at least one trigger field
  (`prefill_trigger_tokens=[-1]` / `generate_trigger_tokens=[-1]` for
  global application) — triggerless configs are rejected instead of
  silently steering nothing.
- Compiled-vs-eager outputs differ by kernel numerics; tests compare
  behavior via the steering trace (`VLLM_STEER_TRACE_DIR`), not bytes.
- The capture package's canonical import path is `vllm.capture`;
  `vllm.hidden_states` is a backward-compatibility alias. Tests import
  the alias on purpose so it stays covered.
