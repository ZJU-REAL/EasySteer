<div align="center">
<h3>
    <img src="figures/logo.png" width="50%"><br>
    A Unified Framework for High-Performance and Extensible LLM Steering
</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-REAL/EasySteer?style=social)](https://github.com/ZJU-REAL/EasySteer/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/commits/main)
[![GitHub](https://img.shields.io/github/license/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2509.25175-b31b1b.svg)](https://arxiv.org/abs/2509.25175)
[![Docker](https://img.shields.io/badge/docker-v0.17.1-orange)](https://hub.docker.com/r/xuhaolei/easysteer/tags)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Lite%20Demo-blue)](https://huggingface.co/spaces/zjuxhl/EasySteer)
[![YouTube](https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=3rRGzZmhrXg)
[![Jiqizhixin](https://img.shields.io/badge/机器之心-Report-blue)](https://mp.weixin.qq.com/s/dxuJHvXOfzA1euvFUPN_vg)

\[ English | [中文](README_zh.md) \]

**[Documentation](https://zju-real.github.io/EasySteer/latest/)** — installation, guides, API reference, replications
</div>

👋 Join our [WeChat](figures/wechat.png) user group. If the QR code has expired, please contact me. (๑•̀ㅂ•́)و✧

<a id="news"></a>
## News 🔥

- [2026/08/03] Migrated to vLLM v0.26.0 (V2 model runner): v2 steering API, redesigned hidden-state capture (select clauses, labeled rows, per-request capture), and our new [documentation site](https://zju-real.github.io/EasySteer/latest/)
- [2026/04/06] [Seeing but Not Thinking: Routing Distraction in Multimodal Mixture-of-Experts](https://arxiv.org/abs/2604.08541) — work built on EasySteer — accepted to the ACL 2026 main conference 🎉
- [2026/03/31] Initial support for vLLM v0.17.1, with server-level steering and CUDA graph support
- [2026/02/16] We've launched an [Lite Demo](https://huggingface.co/spaces/zjuxhl/EasySteer) on Hugging Face Spaces for quick test. For the full-featured version, please refer to the [Web demo docs](https://zju-real.github.io/EasySteer/latest/user-guide/web-demo/).
- [2026/02/15] We've added OpenAI-compatible API support for steering vectors
- [2026/01/11] We’ve adapted EasySteer for vLLM v0.13.0
- [2025/10/31] We’ve adapted EasySteer for vLLM v1 engine.
- [2025/10/10] We’ve adapted EasySteer for the VLMs.
- [2025/09/29] We’ve released our paper.
- [2025/09/28] We’ve open-sourced the code of EasySteer  — feel free to try it out!

## Awesome Work with EasySteer & PRs
- [2026/02/04] Internalizing LLM Reasoning via Discovery and Replay of Latent Actions
[Repository](https://github.com/sznnzs/LLM-Latent-Action)
- [2025/11/23] SHARP: Steering Hallucination in LVLMs via Representation Engineering (EMNLP2025 Main)
[Replication Code](replications/sharp/)

## About

Built on vLLM, EasySteer is a unified framework for high-performance LLM steering: it applies steering vectors — directions in a model's hidden-state space — during inference to shift model behavior without changing model weights, at serving speed. The current release tracks vLLM v0.26.0 on the V2 model runner, with continuous batching, prefix-cache-compatible steering, CUDA-graph support, the declarative v2 steering API (`SteeringSpec`/`ApplySpec`), and a redesigned hidden-state capture pipeline (source-side selection, labeled rows, per-request capture).

- **High Performance**: 10.8-22.3× faster than existing frameworks through vLLM integration
- **Modular Design**: Pluggable interfaces for custom steering algorithms without modifying core code
- **Fine-Grained Control**: Token-level, position-specific, and multi-vector steering capabilities
- **Ready-to-Use**: Pre-computed steering vectors for 8 domains (safety, reasoning, knowledge, etc.)
- **Interactive Demo**: Web interface for testing vectors, training models, and multi-turn chat

## Components

| Component | What it is | Docs |
|---|---|---|
| `vllm-steer/` | vLLM fork with the steering engine (`vllm.steer_vectors`) | [Steering guide](https://zju-real.github.io/EasySteer/latest/user-guide/steering/) |
| `easysteer.hidden_states` | Capture hidden states / MoE router logits from a running engine | [Capture guide](https://zju-real.github.io/EasySteer/latest/user-guide/hidden-state-capture/) |
| `easysteer.steer` | Extract steering vectors from hidden states (analysis-based) | [Extraction guide](https://zju-real.github.io/EasySteer/latest/user-guide/extracting-vectors/) |
| `easysteer.reft` | Train parameterized interventions on frozen models (learning-based) | [ReFT training](https://zju-real.github.io/EasySteer/latest/user-guide/reft-training/) |
| `frontend/` | Web UI for interactive steering experiments | [Web demo](https://zju-real.github.io/EasySteer/latest/user-guide/web-demo/) |
| `replications/` | Reproductions of published steering papers | [Replications](https://zju-real.github.io/EasySteer/latest/replications/) |

## Getting Started

### Installation

Quick install (prebuilt wheel + fork overlay) — installs the official vLLM wheel and applies the fork's Python changes on top, with no compilation and no editable checkouts:

```bash
conda create -n easysteer python=3.12 -y
conda activate easysteer

# Official vLLM wheel, then overlay the fork's Python files
pip install vllm==0.26.0
git clone --depth 1 https://github.com/ZJU-REAL/EasySteer-vllm-v1.git
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
rsync -a EasySteer-vllm-v1/vllm/ "$VLLM_DIR"/

# EasySteer package
git clone https://github.com/ZJU-REAL/EasySteer.git
pip install ./EasySteer
```

Note that reinstalling or upgrading `vllm` reverts the overlay; re-apply the `rsync` step afterwards.

Development install (editable, recommended for ongoing work):

```bash
conda create -n easysteer python=3.12 -y
conda activate easysteer

# Clone the repository (with submodules)
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

# Install with pre-compiled version (recommended)
# EasySteer tracks the vLLM v0.26.0 release commit; pin it so the kernels match.
export VLLM_PRECOMPILED_WHEEL_COMMIT=568afb3a13806beb53bb2e6bd518269357b237c0
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Install EasySteer
cd ..
pip install --editable .
```

For full details, build-from-source, and Docker, see the [installation guide](https://zju-real.github.io/EasySteer/latest/getting-started/installation/).

### A 30-Second Example

```python
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

# enable_steer_vector=True turns on steering; without it, behaves like regular vLLM
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", enable_steer_vector=True, enforce_eager=True)

def happy_steering(scale):
    # Which vector, how strongly, on which layers, and where it applies
    return SteeringSpec(vectors=[VectorSpec(
        source="vectors/happy_diffmean.gguf",
        scale=scale,
        layers=list(range(10, 26)),
        apply=ApplySpec(phases=["prompt", "generation"]),
    )])

text = "<|im_start|>user\nAlice's dog has passed away. Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

baseline = llm.generate(text, steering=happy_steering(0.0), sampling_params=sampling_params)
happy = llm.generate(text, steering=happy_steering(2.0), sampling_params=sampling_params)

print(baseline[0].outputs[0].text)  # ordinary condolences
print(happy[0].outputs[0].text)     # conspicuously upbeat
```

Full walkthrough (including where the vector comes from): [Quickstart](https://zju-real.github.io/EasySteer/latest/getting-started/quickstart/).

### Going Further

- **Steering spec language** (phases, positions, multi-vector, conflict policies, CUDA graphs): [Steering guide](https://zju-real.github.io/EasySteer/latest/user-guide/steering/)
- **OpenAI-compatible HTTP server** (per-request and server-level steering): [Server guide](https://zju-real.github.io/EasySteer/latest/user-guide/openai-server/)
- **Capturing hidden states** and MoE router logits: [Capture guide](https://zju-real.github.io/EasySteer/latest/user-guide/hidden-state-capture/)
- **Extracting vectors** (DiffMean, PCA, LAT, linear probe, SAE): [Extraction guide](https://zju-real.github.io/EasySteer/latest/user-guide/extracting-vectors/)
- **Training interventions** (ReFT / LoReFT / LM-Steer): [ReFT training](https://zju-real.github.io/EasySteer/latest/user-guide/reft-training/)
- **API reference**: [zju-real.github.io/EasySteer/latest/api-reference/](https://zju-real.github.io/EasySteer/latest/api-reference/)

## Contributing

We welcome replications of your papers, new steering algorithms (two methods to implement), and component-level steers (attention/MLP interfaces are reserved in `vllm-steer/vllm/steer_vectors/models.py`). If you have used EasySteer in your research, reach out and we'll feature your work in [News](#news). See the [contributing guide](https://zju-real.github.io/EasySteer/latest/developer-guide/contributing/) for the algorithm template, module map, and ground rules, and the [testing guide](https://zju-real.github.io/EasySteer/latest/developer-guide/testing/) for the test suites.

## Paper Replications

The [replications](replications) folder reproduces published steering papers with EasySteer — full table with paper titles in the [replication gallery](https://zju-real.github.io/EasySteer/latest/replications/):

| Category | Replications |
|---|---|
| Reasoning | [Thinking Speed](replications/controlingthinkingspeed/) · [Fractional Reasoning](replications/fractreason/) · [Improve Reasoning](replications/improve_reasoning/) · [SEAL](replications/seal/) |
| Safety | [Refusal Direction](replications/refusal_direction/) · [CAST](replications/cast/) |
| Style | [Creative Writing](replications/creative_writing/) · [Steerable Chatbots](replications/steerable_chatbot/) |
| Knowledge & Reality | [SAKE](replications/sake/) · [SAE Entities](replications/sae_entities/) · [SHARP](replications/sharp/) |
| General & Personalization | [LM-Steer](replications/lm_steer/) · [LoReFT](replications/loreft/) · [BiPO](replications/bipo/) |
| MoE | [SteerMoE](replications/steermoe/) |

## Citation

If you use EasySteer for your research, please cite our paper:

```bibtex
@article{xu2025easysteer,
  title={EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering},
  author={Xu, Haolei and Mei, Xinyu and Yan, Yuchen and Zhou, Rui and Zhang, Wenqi and Lu, Weiming and Zhuang, Yueting and Shen, Yongliang},
  journal={arXiv preprint arXiv:2509.25175},
  year={2025}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Usage Statement

LLM steering is dual-use. EasySteer is developed primarily as a research tool for advancing model safety, not for circumventing safeguards: steering should be restricted to legitimate research and safety-enhancing applications, any behavioral modifications must be explicitly disclosed to end users, and all applications must adhere to relevant ethical guidelines and legal frameworks.

## Acknowledgements

We thank the [vLLM](https://github.com/vllm-project/vllm) project for providing the high-performance inference framework, and projects like [pyreft](https://github.com/stanfordnlp/pyreft) for their contributions to the field of representation learning. Related projects: [EasyEdit](https://github.com/zjunlp/EasyEdit) · [pyreft](https://github.com/stanfordnlp/pyreft) · [repeng](https://github.com/vgel/repeng) · [vLLM](https://github.com/vllm-project/vllm)

## Star History

<!-- Rendered daily by .github/workflows/star-history.yml using the vendored
     star-history renderer (.github/actions/star-history); the stargazers API
     requires repo-authorized tokens now, so third-party embeds no longer work. -->
<a href="https://github.com/ZJU-REAL/EasySteer/stargazers">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://zju-real.github.io/EasySteer/star-history-dark.svg">
    <img alt="Star History Chart" src="https://zju-real.github.io/EasySteer/star-history-light.svg">
  </picture>
</a>
