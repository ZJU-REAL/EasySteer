# Test suite

Validation tests live in [`tests/`](https://github.com/ZJU-REAL/EasySteer/tree/main/tests)
and exercise the vLLM fork through the v2 steering API. See `tests/README.md` for the
authoritative version of this page.

## Layout

| Path | Contents |
|---|---|
| `tests/conftest.py` | GPU/trace environment setup; a GPU module declares `ENGINE_KWARGS` at module level and every test in it shares that one engine (`llm` / `trace` fixtures). |
| `tests/helpers.py` | Model/vector path constants (env-overridable), the `steering_spec()` builder, and the `TraceOracle` (exact steered positions from the steering trace). |
| `tests/cpu/` | No GPU: spec validation/translation/fingerprints and the position collector (`test_api_spec.py`), algorithm loaders (`test_loaders.py`), vector-store versioning (`test_store.py`). |
| `tests/e2e/` | Dense-model engine suites, one engine config per module: apply-clause semantics under chunked prefill, slot routing + scale sweep, prefix caching, engine-default steering, golden sentiment, require-preload, piecewise and full-graph compilation, capture. |
| `tests/moe/` | OLMoE/Qwen3-MoE suites: router-logit steering modes and slot routing, compiled MoE, SteerMoE replication, second-architecture smoke. |
| `tests/run_suites.sh` | Runs each GPU module in its own pytest process (vLLM engines do not reliably release GPU state in-process). |
| `tests/verify_steering_correctness.py` | Model/hardware sanity harness (argparse tool, not pytest). |
| `tests/bench_eager_vs_cudagraphs.py` | Decode throughput benchmark. |

## Running

```bash
GPU_ID=2 ./tests/run_suites.sh cpu     # no-GPU suites
GPU_ID=2 ./tests/run_suites.sh dense   # dense e2e suites
GPU_ID=2 ./tests/run_suites.sh moe     # MoE suites
GPU_ID=2 ./tests/run_suites.sh all
```

Key environment variables (defaults in `tests/README.md`): `GPU_ID`,
`STEER_TEST_MODEL`, `STEER_TEST_VECTOR`, `STEER_TEST_MOE_MODEL`, `STEER_TEST_QWEN3`,
`STEER_TEST_EAGER`, `STEER_TEST_TP`, `STEERMOE_PKL`.

## Determinism notes

- Byte-exact cross-run comparisons need identical batch geometry: `ignore_eos=True` and
  `async_scheduling=False`. Batched-vs-sequential equality is never asserted (vLLM
  batch-shape numerics).
- Stored goldens are GPU-model-specific; mechanism-level checks (captured logits,
  steering trace, `num_cached_tokens`) are hardware-robust.
- Compiled-vs-eager outputs differ by kernel numerics; compiled tests compare behavior
  via the steering trace, not bytes.
