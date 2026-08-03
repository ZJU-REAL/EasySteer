# EasySteer

**A unified framework for high-performance and extensible LLM steering, built on vLLM.**

EasySteer applies *steering vectors* — directions in a model's hidden-state space — during
inference to shift model behavior without changing model weights. It extends vLLM's V1
engine so that steering runs at serving speed, with continuous batching, prefix caching,
and CUDA-graph support.

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[Paper (arXiv:2509.25175)](https://arxiv.org/abs/2509.25175){ .md-button }

## Why EasySteer

- **High performance** — 10.8–22.3× faster than existing steering frameworks through
  vLLM integration.
- **One spec, every backend** — the same `SteeringSpec` runs under eager, piecewise, and
  full CUDA-graph execution; backends that cannot run a spec reject it explicitly.
- **Fine-grained control** — token-level, position-specific, phase-aware
  (prompt vs. generation), and multi-vector steering.
- **Modular algorithms** — direct addition, linear maps, LoReFT, LM-Steer,
  projection-based erase/replace, MoE router steering; new algorithms plug in with two
  methods.
- **Full research loop** — capture hidden states, extract vectors (DiffMean, PCA, LAT,
  linear probe, SAE), train interventions (ReFT), and serve them, all in one repo.

## The pieces

| Component | What it is |
|---|---|
| `vllm-steer/` | Fork of vLLM with the steering engine (`vllm.steer_vectors`) and hidden-state capture |
| `easysteer.hidden_states` | Capture hidden states / MoE router logits from a running vLLM engine |
| `easysteer.steer` | Extract steering vectors from captured hidden states (analysis-based) |
| `easysteer.reft` | Train parameterized interventions on frozen models (learning-based) |
| `frontend/` | Web UI for interactive steering experiments |
| `replications/` | Notebook reproductions of published steering papers |

## A 30-second look

```python
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", enable_steer_vector=True,
          enforce_eager=True)

spec = SteeringSpec(vectors=[VectorSpec(
    source="vectors/happy_diffmean.gguf",
    scale=2.0,
    layers=list(range(10, 26)),
    apply=ApplySpec(phases=["prompt", "generation"]),
)])

out = llm.generate("Comfort Alice about her dog.",
                   steering=spec,
                   sampling_params=SamplingParams(max_tokens=128))
```

See the [Quickstart](getting-started/quickstart.md) for the full example, and the
[Steering guide](user-guide/steering.md) for the complete spec language.

## Citation

```bibtex
@article{xu2025easysteer,
  title={EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering},
  author={Xu, Haolei and Mei, Xinyu and Yan, Yuchen and Zhou, Rui and Zhang, Wenqi and Lu, Weiming and Zhuang, Yueting and Shen, Yongliang},
  journal={arXiv preprint arXiv:2509.25175},
  year={2025}
}
```

!!! note "Responsible use"
    Steering is dual-use. EasySteer is a research tool for model safety and
    controllability; behavioral modifications must be disclosed to end users and comply
    with applicable guidelines and law.
