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
- `verify_steering_correctness.py` — model/hardware sanity harness (argparse)

MoE gate steering + router-logit capture (OLMoE-1B-7B):
- `test_moe_modes.py` — activate/deactivate semantics, deprecated aliases, mixed configs, no-file requests, trigger positions
- `test_moe_slot_routing.py` — per-request MoE routing in mixed batches
- `test_steermoe_olmoe.py` — SteerMoE (arXiv:2509.09660) end-to-end replication checks
- `test_moe_compiled.py` — MoE gate steering under piecewise CUDA graphs (trace-verified)
- `test_qwen3moe_smoke.py` — second architecture (internal-router gate path)

Capture streams (dense):
- `test_capture_hidden_states.py` — hook-based capture: layer subsets, reductions, budgets, dtypes, legacy RPC shims

CPU-only units:
- `test_mask_equivalence.py` — 5000-case trigger-mask fuzz vs the legacy collector (`_legacy_position_collector.py`)
- `test_store_reload_unit.py` — VectorStore file-version reload semantics

Benchmarks:
- `bench_eager_vs_cudagraphs.py` — decode throughput, eager vs CUDA graphs

## Notes

- Steering requires `enable_prefix_caching=False`; capture streams require
  `enforce_eager=True`.
- Byte-exact cross-run comparisons need identical batch geometry: use
  `ignore_eos=True` and pin `async_scheduling=False` (async admission makes
  prefill co-batching timing-dependent).
- Stored goldens (`golden.txt` comparisons in the sentiment and
  server-slot tests) are GPU-model-specific: kernel numerics differ
  across GPU models, so compare only on the model that recorded the
  golden. Behavior-level checks (does the output change/flip) can also
  be hardware-sensitive; mechanism-level checks (captured logits,
  steering trace) are not.
- Compiled-vs-eager outputs differ by kernel numerics; tests compare
  behavior via the steering trace (`VLLM_STEER_TRACE_DIR`), not bytes.
