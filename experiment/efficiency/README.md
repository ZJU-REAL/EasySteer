# Efficiency benchmarks

Greedy generation throughput on DeepSeek-R1-Distill-Qwen-1.5B over the
same MATH prompts (`../math/math_train_1000.json`), comparing EasySteer
against HF-transformers-based steering frameworks.

Steering is configured at **scale 0** everywhere, so all frameworks
generate identical text and only the steering-path overhead differs.

Setup follows the EasySteer paper (Section 5.1): steering configurations
are single-layer, all-layer (28 layers), and multi-vector (three
sequential vectors on all layers); the framework comparison uses the
all-layer configuration; sequence-length settings are 128 and 2048 max
tokens; metrics are FTL / TPS / TTLT.

```bash
# EasySteer / vLLM (continuous batching)
python bench_vllm.py --mode baseline
python bench_vllm.py --mode single_layer
python bench_vllm.py --mode all_layer
python bench_vllm.py --mode multi_vector
python bench_vllm.py --mode all_layer --batch 256 --max-tokens 2048

# CUDA graphs (paper numbers are eager). Compiled engines default to
# full-graph steering (the kernel is captured into the graph; graph-safe
# configs only); --graph-mode piecewise instead splits the graph at
# every steered layer and supports all algorithms.
python bench_vllm.py --mode all_layer --cudagraph
python bench_vllm.py --mode all_layer --cudagraph --graph-mode piecewise

# pyreft (HF transformers, all-layer zeroed LoReFT; paper batch: 256)
python bench_pyreft.py
python bench_pyreft.py --batch 256

# repeng (HF transformers, all-layer strength 0; paper batch: 64)
python bench_repeng.py
python bench_repeng.py --batch 64
```

Sequential mode times 10 single-prompt requests; `--batch N` times one
batched generate over N prompts. Run one benchmark per GPU at a time.
