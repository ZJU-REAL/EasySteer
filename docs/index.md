# EasySteer

**Capture representations, build interventions, and steer LLM inference with vLLM.**

EasySteer changes intermediate model activations during inference while keeping
model weights fixed. It supports decoder hidden states, attention head outputs,
and MoE router logits through a shared request API. The current engine is based
on vLLM **0.29.0**, with V2 GPU model-runner integration, continuous batching,
prefix caching, and CUDA graph support.

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[Paper (arXiv:2509.25175)](https://arxiv.org/abs/2509.25175){ .md-button }

## Start with your task

| I want to… | Start here |
|---|---|
| Try a bundled steering vector | [Installation](getting-started/installation.md) → [Quickstart](getting-started/quickstart.md) |
| Control layers, tokens, and multiple interventions | [Steering requests](user-guide/steering.md) |
| Capture model activations and learn a direction | [Capture](user-guide/hidden-state-capture.md) → [Extracting vectors](user-guide/extracting-vectors.md) |
| Intervene on attention heads | [Attention and ITI](user-guide/attention.md) |
| Train a ReFT intervention | [ReFT training](user-guide/reft-training.md) |
| Deploy an API or demo | [OpenAI server](user-guide/openai-server.md) → [Web demos](user-guide/web-demo.md) |
| Choose graph, caching, and capacity settings | [Performance](user-guide/performance.md) · [Engine arguments](api-reference/engine-configuration.md) |
| Reproduce a paper | [Replications](replications/index.md) |

## How the pieces fit

| Component | What it is |
|---|---|
| `vllm-steer/` | vLLM fork exposing `vllm.steer_vectors` and `vllm.capture` |
| `easysteer.hidden_states` | Capture labelled hidden states, attention head outputs, and MoE router logits |
| `easysteer.steer` | Extract steering vectors from captured hidden states (analysis-based) |
| `easysteer.vectors` | Convert files or extraction results into common inference payloads |
| `easysteer.reft` | Train parameterized interventions on frozen models (learning-based) |
| `frontend/`, `hf-space/` | Full research frontend and lightweight hosted demo |
| `replications/` | Notebook reproductions of published steering papers |

## A steering request

```python
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enable_steer_vector=True,
    steer_algorithms=["direct"],
)

spec = SteeringSpec(vectors=[VectorSpec(
    source="vectors/happy_diffmean.gguf",
    scale=2.0,
    layers=list(range(10, 24)),
    apply=ApplySpec(prompt="all", generation="all"),
)])

messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "Alice's dog has passed away. Please comfort her."},
]
prompt = {"prompt_token_ids": llm.get_tokenizer().apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
)}
outputs = llm.generate(
    prompt,
    steering=spec,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=128),
)
print(outputs[0].outputs[0].text)
```

Run from the repository root after installation. The
[quickstart](getting-started/quickstart.md) adds the baseline comparison.
The same spec works through the HTTP API; execution mode is selected from the
engine's declared workload. For accepted formats and graph conditions, use the
[algorithm reference](api-reference/algorithms.md).

The [EasySteer paper](https://arxiv.org/abs/2509.25175) reports 10.8–22.3×
speedups over its compared steering frameworks. Use the
[performance guide](user-guide/performance.md) to evaluate the current engine
on your model and hardware.

## Citation

```bibtex
@article{xu2025easysteer,
  title={EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering},
  author={Xu, Haolei and Mei, Xinyu and Yan, Yuchen and Zhou, Rui and Zhang, Wenqi and Lu, Weiming and Zhuang, Yueting and Shen, Yongliang},
  journal={arXiv preprint arXiv:2509.25175},
  year={2025}
}
```
