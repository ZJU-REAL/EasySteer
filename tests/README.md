# EasySteer validation tests

Pytest suites for EasySteer's vLLM fork (`vllm-steer/`), all using the
v2 steering API (`SteeringSpec` / `VectorSpec` / `ApplySpec`, see
`../docs/design/STEERING_API_V2.md`).

## Layout

- `conftest.py` — GPU/trace environment setup and the engine fixture:
  a GPU module declares `ENGINE_KWARGS` at module level and every test
  in it shares that one engine (`llm` / `trace` fixtures).
- `helpers.py` — model/vector path constants (env-overridable), the
  `steering_spec()` builder, and the `TraceOracle` (exact steered
  absolute positions from the steering trace).
- `cpu/` — no GPU: spec validation/translation/fingerprints and the
  position collector (`test_api_spec.py`), algorithm loaders
  (`test_loaders.py`), vector-store versioning (`test_store.py`).
- `e2e/` — dense-model engine suites, one engine config per module:
  apply-clause semantics under chunked prefill, slot routing + scale
  sweep, prefix caching, engine-default steering, golden sentiment,
  require-preload, piecewise and full-graph compilation, capture.
- `moe/` — OLMoE/Qwen3-MoE suites: router-logit steering modes and slot
  routing, compiled MoE, SteerMoE replication, second-architecture
  smoke.
- `run_suites.sh` — runs each GPU module in its own pytest process
  (vLLM engines do not reliably release GPU state in-process):
  `GPU_ID=2 ./run_suites.sh [cpu|dense|moe|all]`.
- `verify_steering_correctness.py` — model/hardware sanity harness
  (argparse tool, not a pytest suite).
- `bench_eager_vs_cudagraphs.py` — decode throughput benchmark.

## Configuration

| Env var | Meaning | Default |
| --- | --- | --- |
| `GPU_ID` | GPU to run on (`CUDA_VISIBLE_DEVICES`) | `0` |
| `STEER_TEST_MODEL` | dense test model (Qwen2.5-1.5B-Instruct) | lab path |
| `STEER_TEST_VECTOR` | steering vector (gguf) for the dense tests | `~/EasySteer/vectors/happy_diffmean.gguf` |
| `STEER_TEST_MOE_MODEL` | MoE test model (OLMoE-1B-7B-0125-Instruct) | `~/models/OLMoE-1B-7B-0125-Instruct` |
| `STEER_TEST_QWEN3` | Qwen3-MoE smoke-test model | lab path |
| `STEER_TEST_EAGER` | `1` eager / `0` compiled (where supported) | per suite |
| `STEER_TEST_TP` | tensor parallel size | `1` |
| `STEERMOE_PKL` | SteerMoE released rankings pickle (optional) | cwd |

## Notes

- Steering supports prefix caching (block hashes are keyed by the
  steering config fingerprint; engine-default mode salts every hash and
  spec updates reset the cache) and chunked prefill. Capture requires
  `enforce_eager=True` and `enable_prefix_caching=False` (cache-hit
  tokens are never recomputed, so they cannot be captured).
- Byte-exact cross-run comparisons need identical batch geometry: use
  `ignore_eos=True` and pin `async_scheduling=False`. Batched-vs-
  sequential equality is NOT asserted anywhere (vLLM batch-shape
  numerics); determinism and anchor checks gate instead.
- Stored goldens (`test_golden_sentiment.py`) are GPU-model-specific:
  compare only on the GPU model that recorded them (currently RTX PRO
  5000, bf16 vectors). Mechanism-level checks (captured logits,
  steering trace, `num_cached_tokens`) are hardware-robust.
- Compiled-vs-eager outputs differ by kernel numerics; compiled tests
  compare behavior via the steering trace, not bytes.
- The capture package's import path is `vllm.capture`.
