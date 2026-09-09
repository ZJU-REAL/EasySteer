# EasySteer validation tests

These suites exercise this checkout's vLLM fork (`vllm-steer/`) through
`SteeringSpec`, `VectorSpec`, and `ApplySpec`. See the
[steering guide](../docs/user-guide/steering.md) for the public API.

## Layout

| Path | Purpose |
| --- | --- |
| `cpu/` | Spec validation, request translation, position selection, payload loading and content caching, management routes, graph kernels and declarations. |
| `kernels/` | Fixed-input CUDA-graph replay and row-isolation checks; no model checkpoint. |
| `e2e/` | Dense-model steering, request isolation, prefix caching, graph modes, hidden-state capture and OpenAI serving. |
| `moe/` | Router-logit steering, compiled MoE, SteerMoE replication and Qwen3-MoE smoke tests. |
| `conftest.py` | Selected-test trace setup, engine startup timing and shared engine profiles. |
| `helpers.py` | Environment-configured model/vector paths, `steering_spec()` and exact steering-position trace checks. |
| `run_suites.sh`, `run_process.py` | Isolated engine groups, bounded child-process cleanup and run artifacts. |
| `verify_steering_correctness.py` | Optional output/logprob diagnostic, with one engine per child process. |
| `bench_eager_vs_cudagraphs.py` | HTTP throughput comparison across requested concurrency levels in eager and in-graph modes. |

## Running

Use an environment with this checkout's fork, EasySteer dependencies, and
`pytest` installed. The runner does not install packages or select a model.
Run commands from the repository root:

```bash
export STEER_TEST_PYTHON=/path/to/environment/bin/python
./tests/run_suites.sh cpu

# This checkpoint matches the bundled happy steering vector.
export STEER_TEST_MODEL=/path/to/Qwen2.5-1.5B-Instruct
GPU_ID=0 ./tests/run_suites.sh baseline
```

| Group | Selection |
| --- | --- |
| `cpu` | All CPU suites; no model variables or GPU required. |
| `kernels` | Fixed-input GPU graph-kernel tests; no model variables required. |
| `baseline` | CPU suites plus the compact GPU selection below. |
| `dense` | Dense-model e2e suites, including HTTP serving, additional graph families and payloads. |
| `moe` | MoE suites; also requires `STEER_TEST_MOE_MODEL` and `STEER_TEST_QWEN3`. |
| `moe-core` | OLMoE eager, split and fullgraph tests; requires only `STEER_TEST_MOE_MODEL`. |
| `extended` | 51-scale batched sweep with sequential variation checks; requires `STEER_TEST_MODEL`. |
| `all` | `cpu`, `kernels`, `dense` and `moe`; requires all three model variables. |
| `golden` | Optional recorded-text comparison; requires `STEER_TEST_GOLDEN` and `STEER_TEST_MODEL`. Excluded from every other group. |

Prefer complete local model directories. An explicitly supplied Hugging Face
ID is passed to vLLM and may download missing files. Direct pytest callers must
set the same model variables. Dense suites assume Qwen2.5-1.5B layer indices,
vocabulary and hidden size; changing the model path does not make them tests
for an arbitrary architecture.

For HTTP-only checks, set `STEER_TEST_SERVER_URL=http://127.0.0.1:8017`
(without `/v1`) and `STEER_TEST_SERVED_MODEL` to reuse an existing service:

```bash
STEER_TEST_SERVER_URL=http://127.0.0.1:8017 \
STEER_TEST_SERVED_MODEL=easysteer-qwen2.5-1.5b \
"$STEER_TEST_PYTHON" -m pytest tests/e2e/test_openai_server.py -q
```

The service must use the test checkpoint, declare only `direct`, and have no
default steering. `STEER_TEST_VECTOR` must identify a file accessible to it.
Reuse checks health, skips the startup-only test, and leaves the process running;
the management test sets and then clears a default. Without these variables the
suite starts and stops its own service as before.

### Compact baseline

`baseline` selects the following tests, in separate GPU processes:

| Selection | Checks |
| --- | --- |
| `cpu/` | Schema, loaders, selection, storage and graph-related unit tests. |
| `kernels/test_graph_additive.py` | Fixed-input graph replay, additive no-op and row isolation. |
| `e2e/test_vanilla_parity.py` | Unsteered traffic with steering enabled versus a vanilla engine. |
| `e2e/test_apply_semantics.py` | Exact eager steering positions and chunked-prefill selectors. |
| `e2e/test_routing.py::TestMultiVector` | Nonzero steering, multi-vector routing and request isolation. |
| `e2e/test_prefix_cache.py` | Steering fingerprint isolation and cache reuse. |
| `e2e/test_piecewise.py` | Split graph execution. |
| `e2e/test_trigger_positions.py` | In-graph prompt and generation position steering. |
| `e2e/test_capture_unified.py` and `e2e/test_server_steering.py` | Capture and default/override/off cache behavior on one compiled engine. |

The baseline excludes the scale sweep, recorded hardware-specific text,
MoE checkpoints and additional HTTP suites. Run the wider groups when those
paths change. A baseline pass does not establish a performance improvement or
compatibility with every supported model and parallelism configuration.

### Comparing changes

Retain logs and results in a fresh directory for each run:

```bash
STEER_TEST_RESULTS_DIR=.local/test-results/before ./tests/run_suites.sh baseline
# After applying the change, use a different directory:
STEER_TEST_RESULTS_DIR=.local/test-results/after ./tests/run_suites.sh baseline
```

The runner writes per-group stdout/stderr, JUnit XML and `summary.tsv`,
continues after failures and exits nonzero if any suite fails. It refuses to
overwrite an existing result directory. Without `STEER_TEST_RESULTS_DIR`,
pytest output goes to the terminal and HTTP startup logs use pytest's temporary
directory. With a results directory, `.server.log` and `.undeclared.log` preserve
HTTP startup output; `.engines.json` records shared-engine/server startup time;
`.process.json` records total process and cleanup time. Pytest's duration report
separates fixture setup, calls and teardown. JUnit includes captured stdout and
stderr for passing tests too, preserving engine startup logs. Cold JIT compilation belongs in the
startup measurement; label whether caches were already populated when comparing.

Different models, graph modes and engine settings run in separate processes.
`test_moe.py` and `test_steermoe.py` explicitly share the `olmoe_eager` profile;
the fixture checks that their engine kwargs match. The runner waits for its
pytest process, then terminates any remaining workers in that process group,
allowing 10 seconds before a forced stop and a final 5-second check. It never
waits for unrelated GPU processes or signals them. A remaining live worker
makes the group fail.

Trace is enabled before engine startup for tests using the `trace` fixture and
modules declaring `STEER_TEST_TRACE = True` for direct trace reads. Other groups
avoid trace synchronization and JSONL output. An explicitly supplied
`VLLM_STEER_TRACE_DIR` still enables tracing for debugging.

The default dense sweep uses scales 0, 1, 2 and 5; `extended` uses 51 scales.
The sequential reference stops once scale 0 is checked and the required number
of distinct outputs is reached. The batched sweep always checks every scale,
including exact trace-based slot isolation and capacity checks. Generation
budgets are at most 576 tokens for the default sweep and 6592 for `extended`.
The extended sweep is excluded from `all`; run it for sweep or stress validation:

```bash
GPU_ID=0 ./tests/run_suites.sh extended
# Equivalent direct selection:
python -m pytest tests/e2e/test_routing.py::TestScaleSweep --steer-extended
```

Record the source revisions and local changes, model revision, vector checksum,
Python/PyTorch/vLLM versions, CUDA runtime, GPU/driver, engine settings and
sampling parameters alongside your results. Keep machine paths, logs and
before/after development records in ignored local storage; they are not public
project documentation. Compare failures against the unchanged baseline before
classifying them as regressions, and inspect skips as well as failures.

## Configuration

| Env var | Meaning | Default |
| --- | --- | --- |
| `STEER_TEST_PYTHON` | Python executable for every pytest process | `python` from the active environment |
| `STEER_TEST_RESULTS_DIR` | Fresh directory for logs, JUnit, startup/process timings and `summary.tsv` | no persistent artifacts |
| `STEER_TEST_GPU_PAUSE` | Optional extra seconds after process-group cleanup | `0` |
| `GPU_ID` | Device index or UUID exposed through `CUDA_VISIBLE_DEVICES` | `0` |
| `STEER_TEST_MODEL` | Dense test model (Qwen2.5-1.5B-Instruct) | required for `baseline`, `dense`, `extended`, `golden`, `all` |
| `STEER_TEST_VECTOR` | Dense steering vector | this checkout's `vectors/happy_diffmean.gguf` |
| `STEER_TEST_MOE_MODEL` | OLMoE-1B-7B-0125-Instruct checkpoint | required for `moe-core`, `moe`, `all` |
| `STEER_TEST_QWEN3` | Qwen3-30B-A3B checkpoint | required for `moe`, `all` |
| `STEER_TEST_GOLDEN` | Explicit recorded-text JSON file | unset; opt-in only |
| `STEER_TEST_EAGER` | `1` eager / `0` compiled (where supported) | per suite |
| `STEER_TEST_TP` | Tensor parallel size | `1` |
| `STEERMOE_PKL` | SteerMoE released rankings pickle | cwd |

### Optional recorded-text comparison

`golden` compares one complete completion with the `output_text` field in a
JSON record; it does not search for the completion inside an old console log.
The record must also contain `metadata` matching the test's GPU name,
PyTorch/vLLM/CUDA versions, model identifier, vector SHA-256, dtype, steering
spec (excluding the machine-specific vector path), prompt, sampling and engine
settings. The test reports the expected metadata and fails before constructing
an engine if these differ. Record the reference output separately on the
intended environment with those exact settings, and preserve the model's
revision alongside the record; a path alone does not identify model contents.

```bash
STEER_TEST_GOLDEN=/path/to/reference.json GPU_ID=0 ./tests/run_suites.sh golden
```

The test uses the README's Alice prompt, direct steering at scale 2 on layers
10–25, and no normalization; generation settings are pinned in
`e2e/test_golden_sentiment.py`.

## Comparing results

Use temperature 0 to reduce sampling variation, with matching batch settings
and generation lengths for comparisons. Small wording differences are normal;
repeat the control when interpreting a text mismatch. Steering positions,
routing and tensor checks retain their own correctness assertions.

The optional diagnostic compares logprobs along shared token prefixes. The
benchmark fixes the requested generation length and measures end-to-end HTTP
throughput, including prefill, using the actual output-token count.

## Coverage limits

- Capture uses `vllm.capture`. Request admission bypasses prefix-cache reads
  when the effective selection needs otherwise skipped prompt rows; writes
  remain enabled and caller salts are preserved. Starting capture after a
  request has already reused selected rows is reported as incomplete at fetch.
  Separate FULL capture graphs require a single worker, no speculative decoding
  or LoRA, and steering disabled or `in_graph`; other selected steps run eagerly.
- Steering with KV-transfer rejection/reconstruction remains a known risk:
  the reconstructed `EngineCoreRequest` may omit steering state. The baseline
  does not cover this path.
- Generation-position selectors with speculative decoding and restoration of
  steering after preemption need dedicated end-to-end validation. Decode-step
  indexing and the prompt/generation boundary during resumed prefill must be
  checked in those dedicated tests.

Current coverage for defaults, overrides, explicit disabling and runtime updates
lives in `e2e/test_server_steering.py`, which shares the compiled capture engine
with `e2e/test_capture_unified.py`, and `e2e/test_openai_server.py`. CPU tests check
that admitted requests retain their configuration when the default changes;
the model tests verify prefix-cache separation and reuse across those changes.
Graph behavior is
covered by `e2e/test_fullgraph*.py`, `e2e/test_piecewise.py` and CPU graph tests.
The former vLLM `basic_correctness` steering demos and tests used removed API
fields and have been retired in favor of these maintained suites.
