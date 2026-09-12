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
| Baseline | 6,923.93 | 10,491.27 | 13,712.53 | **15,453.77** |
| Single layer (layer 20) | 6,633.48 | 9,863.54 | 12,067.68 | **15,124.27** |
| All layers (28 layers) | 5,851.63 | 8,768.30 | 11,036.08 | **15,024.35** |
| Multi-vector (3 × 28 layers) | 4,830.55 | 7,586.36 | 9,352.51 | — |
| All layers, 2048 tokens | 6,253.81 | 9,818.66 | 10,871.24 | **11,613.44** |

## Framework comparison

| Framework | 128 tokens | 2048 tokens |
|---|---:|---:|
| EasySteer, in-graph | **15,024.35** | **11,613.44** |
| EasySteer, split | 11,036.08 | 10,871.24 |
| EasySteer, eager (max 512) | 8,768.30 | 9,818.66 |
| EasySteer, eager (max 256) | 5,851.63 | 6,253.81 |
| PyReFT-compatible additive | 1,049.96 | 668.50 (batch 128) |
| repeng | 1,151.37 | 788.22 (batch 128) |

The PyReFT-compatible batch-256 run at 2048 tokens ended with OOM. The
batch-128 result completed all 512 requests.

## Distinct configurations per batch

This in-graph run uses 512 requests and 28 layers. vLLM resolves
max_steer_vectors to 256 for this workload. K is the number of distinct
zero-scale configurations assigned across the batch. Each K value is
measured in a fresh engine process.

| K | 0 | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TPS | 14,938.23 | 14,615.41 | 13,376.36 | 12,853.19 | 11,539.70 | 8,984.34 | 5,642.41 |
