# Efficiency benchmarks

Greedy generation throughput on DeepSeek-R1-Distill-Qwen-1.5B over the
same MATH prompts (`../math/math_train_1000.json`, not committed — see
`../math/README.md`), comparing EasySteer against HF-transformers-based
steering frameworks and across EasySteer's steering execution tiers.

Steering is configured at **scale 0** everywhere, so all frameworks
generate identical text and only the steering-path overhead differs.

Setup follows the EasySteer paper (Section 5.1): steering configurations
are single-layer, all-layer (28 layers), and multi-vector (three
sequential vectors on all layers); 256 prompts submitted in one call;
`max_tokens` 128 (plus an all-layer 2048 variant); metrics are FTL /
TPS / TTLT.

## Results (2026-08-06)

Environment: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB),
driver 580.173.02, torch 2.11.0+cu130, vllm-steer 0.26.0, batch 256,
`max_tokens=128` unless noted. TPS = aggregate output tokens/s.

### EasySteer execution tiers × steering modes

| mode | eager | split (piecewise graphs) | in_graph (full graphs) |
|---|---|---|---|
| baseline (no steering) | 9564 | 13330 | **33217** |
| single_layer | 9375 | 11459 | **29622** |
| all_layer (28 layers) | 8256 | 10438 | **25310** |
| multi_vector (3 × 28 layers) | 6818 | 8663 | 8483¹ |
| all_layer, max_tokens 2048 | 6788 | 9009 | **19286** |

¹ multi-vector configs are not in-graph-admissible; with
`--cudagraph` the declaration auto-resolves them to the split tier.

### Framework comparison (all-layer steering)

| framework | execution | TPS |
|---|---|---|
| EasySteer (in_graph) | vLLM, full CUDA graphs | **25310** |
| EasySteer (split) | vLLM, piecewise CUDA graphs | 10438 |
| EasySteer (eager) | vLLM, no graphs | 8256 |
| pyreft (batch 256) | HF transformers (no CUDA-graph support) | 1025 |
| repeng (batch 64, paper setting) | HF transformers (no CUDA-graph support) | 921 |

### Distinct configurations per batch (in_graph, `max_steer_vectors=256`)

K distinct zero-scale configs round-robined over 256 requests:

| K | 0 | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| TPS | 30633 | 23211 | 19923 | 15851 | 11082 | 8733 | 5620 |

`max_steer_vectors` is a scheduling constraint (identical configs share
a slot; differently-configured requests beyond the capacity queue); see
`bench_capacity_sweep.py` for capacity as the throttle.

## Commands

```bash
# EasySteer / vLLM (continuous batching); add --cudagraph for the
# in-graph tier (auto), --cudagraph --graph-mode split for piecewise.
python bench_vllm.py --mode baseline     --batch 256 --max-tokens 128
python bench_vllm.py --mode single_layer --batch 256 --max-tokens 128
python bench_vllm.py --mode all_layer    --batch 256 --max-tokens 128
python bench_vllm.py --mode multi_vector --batch 256 --max-tokens 128
python bench_vllm.py --mode all_layer    --batch 256 --max-tokens 2048

# HF-transformers frameworks (paper batches; no CUDA-graph support)
python bench_pyreft.py --batch 256
python bench_repeng.py --batch 64

# Mixed steering configurations per batch (K distinct zero-scale
# configs; --distinct-paths gives every config its own vector file)
python bench_multi_config.py --batch 256 --configs 0 1 8 32 64 128 256 \
    --max-steer 256 --cudagraph

# Apples-to-apples tier comparison at identical batch and K sweep
python bench_mode_compare.py --batch 64 --configs 0 1 8 32 \
    --modes eager split in_graph

# max_steer_vectors as a throughput knob (all-distinct workload
# drained through a swept slot capacity)
python bench_capacity_sweep.py --batch 256 --capacities 2 8 32 128 256
```

Run one benchmark per GPU at a time.
