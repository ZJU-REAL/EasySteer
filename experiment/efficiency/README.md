# Efficiency benchmarks

## Latest results

These are the latest v0.29.0 measurements. The model and software versions
differ from the original paper measurements.

Hardware and software:

- NVIDIA RTX A6000, 48 GB, driver 580.105.08
- vLLM 0.29.0 with PyTorch 2.13.0+cu130
- Transformers 5.16.1, repeng 0.4.0, NumPy 2.3.5
- DeepSeek-R1-Distill-Qwen-1.5B in BF16

The workload contains 512 MATH prompts. vLLM processes them in one batch; the
Transformers baselines use batch 256 for 128-token generation and batch 128
for 2048-token generation. Decoding is greedy and each request produces
exactly the requested number of tokens. The prefix cache is cleared after
warmup and probe calls. TPS is generated tokens divided by elapsed generation
time;
TTLT is elapsed time divided by the number of requests.

## EasySteer execution tiers

All steering vectors use the direct algorithm with scale 0. Single-layer
steering targets layer 20, all-layer steering targets 28 layers, and
multi-vector steering uses three vectors on all layers. The resolved execution
modes are NONE, PIECEWISE, and FULL_AND_PIECEWISE for eager, split, and
in-graph execution.

| Mode | Eager (max 256) | Eager (max 512) | Split (max 512) | In-graph (max 512) |
|---|---:|---:|---:|---:|
| Baseline | 6,170.28 | 10,207.12 | 13,216.10 | **14,956.94** |
| Single layer (layer 20) | 5,190.10 | 9,856.90 | 12,871.58 | **14,706.51** |
| All layers (28 layers) | 4,654.24 | 8,716.36 | 10,882.72 | **14,597.28** |
| Multi-vector (3 × 28 layers) | 4,193.97 | 7,565.08 | 9,142.93 | — |
| All layers, 2048 tokens | 5,458.78 | 9,453.38 | 10,979.25 | **11,531.75** |

## Framework comparison

| Framework | 128 tokens | 2048 tokens |
|---|---:|---:|
| EasySteer, in-graph | **14,597.28** | **11,531.75** |
| EasySteer, split | 10,882.72 | 10,979.25 |
| EasySteer, eager (max 512) | 8,716.36 | 9,453.38 |
| EasySteer, eager (max 256) | 4,654.24 | 5,458.78 |
| PyReFT-compatible additive | 1,049.96 | 668.50 (batch 128) |
| repeng | 1,151.37 | 788.22 (batch 128) |

The PyReFT-compatible batch-256 run at 2048 tokens ended with OOM. The
batch-128 result completed all 512 requests.

## Distinct configurations per batch

This in-graph run uses 512 requests, 28 layers, and max_steer_vectors=256.
K is the number of distinct zero-scale configurations assigned across the
batch.

| K | 0 | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TPS | 14,982.30 | 12,632.06 | 14,049.05 | 13,729.54 | 12,156.79 | 10,876.59 | 8,039.90 |
