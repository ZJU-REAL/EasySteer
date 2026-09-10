# Efficiency benchmarks

Measure generation throughput and steering overhead on
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, using the same MATH prompts and
zero-scale control vectors (or zeroed LoReFT parameters). The original
experiment settings follow Section 5.1 of the EasySteer paper.

## Setup

Run the scripts from `experiment/efficiency/` in an installed EasySteer
v0.29.0 environment. Use one benchmark process per GPU.

- `EASYSTEER_MODEL`: the model ID above, or an existing local copy of that model.
- `EASYSTEER_VECTOR`: defaults to the committed SEAL
  `replications/seal/execution_avg_vector.gguf`. It must match the model.
- `EASYSTEER_BENCH_DATA`: defaults to `../math/math_train_1000.json`, a JSON
  list of problem strings. Data is not committed; see `../math/README.md`.

The prompt format remains the experiment's raw R1 reasoning prompt. The
all-layer setting uses this model's 28 layers; the single-layer setting uses
layer 20. Pointing the environment variable at a different architecture is
not a substitute for reproducing this workload.

The Transformers baselines use the bundled `easysteer.reft.pyreft` and the
separately installed `repeng` package. The repeng script retains its NumPy
alias compatibility for the older dependency; it is not needed by EasySteer
or vLLM. Record dependency versions when rerunning either baseline.

## Measurement

Current runs explicitly use BF16 weights, greedy decoding, and fixed output
lengths: `ignore_eos=True` in vLLM and matching `min_new_tokens` /
`max_new_tokens` in Transformers. Metrics count generated token IDs.

- TPS is aggregate generated tokens divided by elapsed wall time.
- TTLT retains the historical label for elapsed wall time divided by the
  number of submitted requests. It is amortized batch time, not the mean
  latency observed by individual requests.
- `bench_vllm.py` additionally reports the wall time of a one-token generate
  call. For a batch this waits for all requests; it is not streaming TTFT.

Model loading, compilation, CUDA graph recording, and an eight-token warmup
are outside the timer. Steady-state rows warm up the same steering workload,
including its payloads. Transformers measurements synchronize CUDA before
and after timing. vLLM's blocking `generate()` returns completed outputs.

The vLLM comparisons disable prefix caching and chunked prefill consistently,
so warmup cannot turn a measured prompt into a prefix-cache hit. Eager,
split, and in-graph execution remain explicit experimental controls; these
settings are not recommendations for serving. The engine prints its resolved
graph mode before measurement.

`--distinct-paths` is the exception to payload warmup: every K gets new vector
paths, and its first (`cold`) row includes loading those paths. The second
(`warm`) row reuses them. Model warmup remains outside both timers. Temporary
vector files are removed after the run.

The framework baselines retain their original intervention settings:
EasySteer adds a zero-scale vector on 28 layers, repeng uses layers 1–27,
and pyreft attaches zeroed rank-4 LoReFT on 28 layers. These compare framework
workloads, not identical intervention kernels. Transformers inputs are
prepared before timing; vLLM timing includes prompt processing in `generate()`.

## Quick rerun

Start with six throughput rows at one batch size and generation length.
Each execution mode loads its own engine and measures an unsteered batch
followed by a batch sharing one all-layer configuration:

```bash
python bench_mode_compare.py --batch 64 --configs 0 1 \
    --max-steer 32 --max-tokens 128 --modes eager split in_graph
```

Add K=8 to the same command when checking mixed request configurations. A
separate multi-vector run checks the sequential-composition path, which
resolves to split execution:

```bash
python bench_vllm.py --mode multi_vector --batch 64 --max-tokens 128 --cudagraph
```

This subset avoids repeating every historical batch, capacity, and output
length. Keep the model, GPU, dependency versions, batch size, and token count
with the resulting measurements.

## Other benchmark commands

```bash
# Original vLLM workload settings; eager is the default control.
# Add --cudagraph for automatic selection, or
# --cudagraph --graph-mode split to select piecewise execution.
python bench_vllm.py --mode baseline     --batch 256 --max-tokens 128
python bench_vllm.py --mode single_layer --batch 256 --max-tokens 128
python bench_vllm.py --mode all_layer    --batch 256 --max-tokens 128
python bench_vllm.py --mode multi_vector --batch 256 --max-tokens 128
python bench_vllm.py --mode all_layer    --batch 256 --max-tokens 2048

# Transformers baselines at their original batch sizes.
python bench_pyreft.py --batch 256 --max-tokens 128
python bench_repeng.py --batch 64 --max-tokens 128

# K configurations sharing a vector file; add --distinct-paths to compare
# first load against reuse with a separate file for every configuration.
python bench_multi_config.py --batch 256 --configs 0 1 8 32 64 128 256 \
    --max-steer 256 --max-tokens 128 --cudagraph

# One distinct configuration per request; sweep the slot capacity.
python bench_capacity_sweep.py --batch 256 --capacities 2 8 32 128 256 \
    --max-tokens 128
```

## Results: v0.29.0 on RTX A6000

Measured on 2026-09-11 (UTC) with one NVIDIA RTX A6000 (48 GB), driver
580.105.08, PyTorch 2.13.0+cu130, and Transformers 5.16.1. The model uses BF16
and its default 131,072-token context. All modes use `OMP_NUM_THREADS=4`.

The comparison used batch 64, 128 output tokens per request, and 32 steering
slots. Every row generated exactly 8,192 tokens. K counts distinct zero-scale
configurations assigned across the batch; K=0 disables steering. Each
configuration targets all 28 layers, with a distinct prompt-position exclusion.

| Execution mode | K=0 | K=1 | K=8 |
|---|---:|---:|---:|
| Eager | 2,115.21 | 1,703.71 | 1,080.25 |
| Split | 3,613.63 | 2,590.09 | 1,359.41 |
| In-graph | 7,685.29 | 7,161.86 | 6,932.30 |

Values are generated tokens/s. The engine resolved these modes to `NONE`,
`PIECEWISE`, and `FULL_AND_PIECEWISE`, respectively. At K=1, in-graph execution
was 4.20× faster than eager in this workload.

A separate three-vector sequential run automatically selected split execution:
8,192 tokens in 5.6870 seconds, or **1,440.48 tokens/s**. Its one-token batch
call took 345.31 ms; this measures the completed batch, not streaming TTFT.

These are single measurements of the current benchmark, using the warmup
procedure above. They compare execution modes within v0.29.0, not performance
against an older release or the paper's different batch sizes. The ten timed
runs took 36.8 seconds in total; the full process took 9.5 minutes including
loading, initialization, compilation, and warmup.
