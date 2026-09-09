# Test suite

Validation tests live in [`tests/`](https://github.com/ZJU-REAL/EasySteer/tree/main/tests)
and exercise the vLLM fork through `SteeringSpec`, `VectorSpec` and `ApplySpec`.
The [test guide](https://github.com/ZJU-REAL/EasySteer/blob/main/tests/README.md)
lists every group, environment variable and the exact compact baseline selection.

## Running

Use an environment with this checkout's fork, EasySteer dependencies and
`pytest` installed. The runner does not install dependencies or select model IDs.
Run from the repository root:

```bash
export STEER_TEST_PYTHON=/path/to/environment/bin/python
./tests/run_suites.sh cpu
export STEER_TEST_MODEL=/path/to/Qwen2.5-1.5B-Instruct
GPU_ID=0 ./tests/run_suites.sh baseline
GPU_ID=0 ./tests/run_suites.sh dense
```

`kernels` runs fixed-input GPU graph-kernel tests without a model checkpoint.
`baseline` includes these, CPU tests and seven model-based GPU selections covering unsteered
parity, nonzero steering and request isolation, exact token positions, prefix
caching, split and in-graph modes, and hidden-state capture on a compiled engine.
Different engine configurations run in separate processes. Identical named
profiles can share one engine: the two OLMoE eager modules use this path.
It excludes the full scale sweep, recorded text, MoE models and additional HTTP
suites. Use `dense` for broader dense-model coverage, including serving.

`moe` requires `STEER_TEST_MOE_MODEL` (OLMoE-1B-7B-0125-Instruct) and
`STEER_TEST_QWEN3` (Qwen3-30B-A3B); `moe-core` only requires the OLMoE model.
`all` combines CPU, kernels, dense and MoE groups. The dense sweep uses four representative
scales; `extended` runs the full 51-scale sweep separately.
Prefer complete local model directories. Explicit Hugging Face IDs may download
missing files. Dense tests assume Qwen2.5-1.5B's architecture and the bundled
`vectors/happy_diffmean.gguf`; an arbitrary model cannot replace it without
adapting the test workload and vector.

The HTTP suite can reuse an existing service with `STEER_TEST_SERVER_URL`
(the base URL without `/v1`) and optional `STEER_TEST_SERVED_MODEL`. It checks
health and leaves the service running. The test service should declare only
`direct` and have no default steering; see the test guide for the command.

## Recording and comparing results

Set `STEER_TEST_RESULTS_DIR` to a fresh directory for per-suite logs, JUnit XML
and `summary.tsv`. Startup and process timing JSON files separate engine boot
from total runtime; pytest reports setup, call and teardown durations. HTTP
startup logs are retained alongside the results. Distinguish cold compilation
from runs using populated caches when comparing times.

The runner continues after failures and returns nonzero if any suite fails.
Each group runs in its own process group; remaining child workers receive a
bounded shutdown before the next group starts. `STEER_TEST_GPU_PAUSE` adds an
optional pause after cleanup (default 0 seconds). Trace is enabled before engine
startup only for selected trace-based tests, unless explicitly requested through
`VLLM_STEER_TRACE_DIR`.

```bash
STEER_TEST_RESULTS_DIR=.local/test-results/before ./tests/run_suites.sh baseline
STEER_TEST_RESULTS_DIR=.local/test-results/after ./tests/run_suites.sh baseline
```

Record source revisions/local changes, model revision, vector checksum,
Python/PyTorch/vLLM versions, CUDA runtime, GPU/driver, engine configuration and
sampling parameters. Keep run logs, server paths and development records in
ignored local storage. Public documentation describes reproducible methods and
limitations; it does not include machine-specific acceptance logs.

## Comparing results

Use matching batch and generation settings when comparing runs. Temperature 0
reduces sampling variation; small wording differences are normal. See the
[test guide](https://github.com/ZJU-REAL/EasySteer/blob/main/tests/README.md)
for control comparisons and result recording.

The `golden` group is opt-in and excluded from `baseline`, `dense` and `all`.
It requires `STEER_TEST_GOLDEN`, a recorded JSON completion with matching
GPU/software/workload metadata. A metadata mismatch fails before constructing
an engine. See the test guide for the record contract.

`tests/verify_steering_correctness.py` collects diagnostic observations with
separate engine processes. It reports logprob differences along shared token
prefixes.
`tests/bench_eager_vs_cudagraphs.py` compares identical HTTP workloads in eager
and in-graph modes. Default, explicit and disabled steering resolve to the same
per-request execution path.
Its throughput includes prefill and HTTP overhead; rerun it to obtain results
for your environment instead of treating historical measurements as current.
Use `--concurrency 1 8` to compare serial and concurrent requests on each engine.
Per-request mode measures unsteered, zero-scale and nonzero workloads separately;
engine-default mode measures its configured steering workload. Warmup uses the
same concurrency as the measured requests, startup time is reported separately,
and tracing is disabled for the benchmark.

## Coverage limits

The baseline does not validate steering state across KV-transfer rejection,
speculative-decoding generation selectors or preemption recovery. The KV
reconstruction path may omit steering state; speculative decode-step indexing
and resumed-prefill prompt/generation boundaries need dedicated validation.
A baseline pass also does not certify every MoE model, parallelism setting,
conditional graph payload or a performance improvement.

Capture uses `vllm.capture`. Eligible single-worker FULL batches use a separate
capture graph; other steps needing rows dispatch eagerly, while empty selections
keep normal execution. `capture_status` exposes actual graph replay and eager
forward counters. Admission skips prefix-cache reads only when a hit could omit
selected prompt rows, preserving normal cache writes. Requests already admitted
before capture starts fail at fetch if selected rows were skipped.
