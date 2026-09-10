<div align="center">
<h3>
    <img src="figures/logo.png" width="50%"><br>
    A Unified Framework for High-Performance and Extensible LLM Steering
</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-REAL/EasySteer?style=social)](https://github.com/ZJU-REAL/EasySteer/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/commits/main)
[![GitHub](https://img.shields.io/github/license/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2509.25175-b31b1b.svg)](https://arxiv.org/abs/2509.25175)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Lite%20Demo-blue)](https://huggingface.co/spaces/zjuxhl/EasySteer)
[![YouTube](https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=3rRGzZmhrXg)
[![Jiqizhixin](https://img.shields.io/badge/机器之心-Report-blue)](https://mp.weixin.qq.com/s/dxuJHvXOfzA1euvFUPN_vg)

\[ [English](README.md) | 中文 \]

**[文档站](https://zju-real.github.io/EasySteer/latest/)** — 安装、使用指南、API 参考、论文复现（文档为英文）
</div>

👋 加入我们的 [微信群](figures/wechat.png)。如果二维码过期了，请联系我。(๑•̀ㅂ•́)و✧

<a id="news"></a>
## 新闻 🔥

- [2026/09/10] 当前版本基于 vLLM v0.29.0（V2 模型运行器），新增注意力头输出捕获（`attention_heads`）、`attention_add` 干预与 [ITI 复现](replications/iti/)。干预、捕获与 Docker 用法见[文档站](https://zju-real.github.io/EasySteer/latest/)。
- [2026/08/22] [EasySteer 论文](https://arxiv.org/abs/2509.25175)被 EMNLP 2026 System Demonstrations 接收 🎉
- [2026/04/06] 基于 EasySteer 的工作 [Seeing but Not Thinking: Routing Distraction in Multimodal Mixture-of-Experts](https://arxiv.org/abs/2604.08541) 被 ACL 2026 主会接收 🎉
- [2026/02/15–16] 上线 [Hugging Face 轻量级 Demo](https://huggingface.co/spaces/zjuxhl/EasySteer) 和 [OpenAI 兼容干预 API](https://zju-real.github.io/EasySteer/latest/user-guide/openai-server/)。完整 Web 界面见[演示文档](https://zju-real.github.io/EasySteer/latest/user-guide/web-demo/)。
- [2025/10/10] 新增视觉语言模型（VLM）支持。
- [2025/09/28–29] 开源 EasySteer 代码并发布[论文](https://arxiv.org/abs/2509.25175)。

## 使用 EasySteer 的优秀工作与 PRs
- [2026/02/04] Internalizing LLM Reasoning via Discovery and Replay of Latent Actions
[仓库地址](https://github.com/sznnzs/LLM-Latent-Action)
- [2025/11/23] SHARP: Steering Hallucination in LVLMs via Representation Engineering (EMNLP2025 Main)
[复现代码](replications/sharp/)

## 关于 EasySteer

EasySteer 是一个基于 vLLM 构建的高性能 LLM 干预（steering）统一框架：它在推理过程中向模型隐状态空间施加干预向量，在不修改模型权重的前提下改变模型行为，并保持推理服务级别的速度。当前版本基于 vLLM v0.29.0（V2 模型运行器），提供连续批处理、兼容前缀缓存的干预、CUDA 图支持、声明式 v2 干预 API（`SteeringSpec`/`ApplySpec`），以及重新设计的隐状态捕获管线（源侧选择、行标签、按请求捕获）。

- **高性能**: 论文实验中，通过对接 vLLM，相比所比较的干预框架实现 10.8-22.3× 的速度提升
- **模块化设计**: 插拔式接口，便于在不改动核心代码的情况下扩展自定义算法
- **细粒度控制**: 支持按 token、按位置、按多向量的精细化干预
- **可即用**: 提供覆盖 8 个领域（安全、推理、知识等）的预计算向量
- **交互式演示**: 提供 Web 界面用于测试向量、训练模型与多轮对话

## 组件一览

| 组件 | 说明 | 文档 |
|---|---|---|
| `vllm-steer/` | 内置干预引擎（`vllm.steer_vectors`）的 vLLM 分支 | [干预指南](https://zju-real.github.io/EasySteer/latest/user-guide/steering/) |
| `easysteer.hidden_states` | 从运行中的引擎捕获隐状态 / MoE 路由 logits | [捕获指南](https://zju-real.github.io/EasySteer/latest/user-guide/hidden-state-capture/) |
| `easysteer.steer` | 从隐状态中提取干预向量（分析式） | [向量提取指南](https://zju-real.github.io/EasySteer/latest/user-guide/extracting-vectors/) |
| `easysteer.reft` | 在冻结模型上训练参数化干预（学习式） | [ReFT 训练](https://zju-real.github.io/EasySteer/latest/user-guide/reft-training/) |
| `frontend/` | 交互式干预实验的 Web 界面 | [Web 演示](https://zju-real.github.io/EasySteer/latest/user-guide/web-demo/) |
| `replications/` | 已发表干预论文的复现 | [复现集](https://zju-real.github.io/EasySteer/latest/replications/) |

## 快速上手

### 安装

快速安装（预编译 wheel + 分支覆盖）——安装官方 vLLM wheel，并将分支的 Python 改动覆盖其上，无需编译、无需可编辑安装：

```bash
conda create -n easysteer python=3.12 -y
conda activate easysteer

# 克隆 EasySteer 及其固定版本的 vLLM 子模块
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer

# 官方 vLLM wheel，然后覆盖固定版本分支的 Python 文件
pip install vllm==0.29.0
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
rsync -a vllm-steer/vllm/ "$VLLM_DIR"/

# EasySteer 包
pip install .
```

注意：重装或升级 `vllm` 会还原该覆盖，需重新执行 `rsync` 一步。

开发安装（可编辑模式，推荐用于持续开发）：

```bash
conda create -n easysteer python=3.12 -y
conda activate easysteer

# 克隆仓库（包含子模块）
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer

# 已有仓库时，在 EasySteer 根目录从这一步开始
git submodule update --init --recursive
cd vllm-steer

# 使用预编译版本安装（推荐）
# EasySteer 适配的是 vLLM v0.29.0 发布时的 commit，请指定该 commit 以匹配预编译内核。
export VLLM_PRECOMPILED_WHEEL_COMMIT=98dff2a81d747d1dba01a47f939f48c3526d4206
VLLM_USE_PRECOMPILED=1 pip install --editable .

# 安装 EasySteer
cd ..
pip install --editable .
```

完整说明、源码编译与 Docker 方案见[安装指南](https://zju-real.github.io/EasySteer/latest/getting-started/installation/)。

### 30 秒示例

请在 EasySteer 仓库根目录运行，以便找到相对路径 `vectors/` 下的向量。

```python
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

# enable_steer_vector=True 开启干预功能；不加则与普通 vLLM 行为一致。
# steer_algorithms 声明请求将使用的干预算法——引擎据此选择能服务它们的最快 CUDA 图集成方式。
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", enable_steer_vector=True,
          steer_algorithms=["direct"])

def happy_steering(scale):
    # 使用哪个向量、强度多大、作用在哪些层、作用在哪些位置
    return SteeringSpec(vectors=[VectorSpec(
        source="vectors/happy_diffmean.gguf",
        scale=scale,
        layers=list(range(10, 24)),
        apply=ApplySpec(prompt="all", generation="all"),
    )])

tokenizer = llm.get_tokenizer()
prompt = {"prompt_token_ids": tokenizer.apply_chat_template(
    [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Alice's dog has passed away. Please comfort her."},
    ],
    tokenize=True,
    add_generation_prompt=True,
)}
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

baseline = llm.generate(prompt, steering=False, sampling_params=sampling_params)
happy = llm.generate(prompt, steering=happy_steering(2.0), sampling_params=sampling_params)

print(baseline[0].outputs[0].text)  # 常规安慰
print(happy[0].outputs[0].text)     # 明显偏"快乐"的输出
```

完整流程（包括这个向量是怎么得到的）见[快速开始](https://zju-real.github.io/EasySteer/latest/getting-started/quickstart/)。

### 进一步了解

- **干预 spec 语言**（phase、位置、多向量、冲突策略、CUDA 图）: [干预指南](https://zju-real.github.io/EasySteer/latest/user-guide/steering/)
- **OpenAI 兼容 HTTP 服务**（按请求与服务端级干预）: [服务指南](https://zju-real.github.io/EasySteer/latest/user-guide/openai-server/)
- **捕获隐状态**与 MoE 路由 logits: [捕获指南](https://zju-real.github.io/EasySteer/latest/user-guide/hidden-state-capture/)
- **提取向量**（DiffMean、PCA、LAT、线性探针、SAE）: [向量提取指南](https://zju-real.github.io/EasySteer/latest/user-guide/extracting-vectors/)
- **训练干预**（ReFT / LoReFT / LM-Steer）: [ReFT 训练](https://zju-real.github.io/EasySteer/latest/user-guide/reft-training/)
- **API 参考**: [zju-real.github.io/EasySteer/latest/api-reference/](https://zju-real.github.io/EasySteer/latest/api-reference/)

## 如何贡献

欢迎通过 PR 复现你的论文、贡献新的干预算法，以及支持更多模型和组件。算法通过继承 `BaseSteerVectorAlgorithm` 扩展，模型适配通过模块发现和干预控制器实现。如果你在研究中使用了 EasySteer，欢迎联系我们，我们很乐意在[新闻](#news)中展示你的工作。扩展示例与模块结构见[贡献指南](https://zju-real.github.io/EasySteer/latest/developer-guide/contributing/)，测试套件见[测试指南](https://zju-real.github.io/EasySteer/latest/developer-guide/testing/)。

## 论文复现

[replications](replications) 目录使用 EasySteer 实现已发表论文的干预方法，各示例的范围见[复现集](https://zju-real.github.io/EasySteer/latest/replications/)：

| 类别 | 复现 |
|---|---|
| 推理 | [Thinking Speed](replications/controlingthinkingspeed/) · [Fractional Reasoning](replications/fractreason/) · [Improve Reasoning](replications/improve_reasoning/) · [SEAL](replications/seal/) |
| 安全 | [Refusal Direction](replications/refusal_direction/) · [CAST](replications/cast/) |
| 风格 | [Creative Writing](replications/creative_writing/) · [Steerable Chatbots](replications/steerable_chatbot/) |
| 知识与真实性 | [SAKE](replications/sake/) · [SAE Entities](replications/sae_entities/) · [SHARP](replications/sharp/) · [ITI](replications/iti/) |
| 通用与个性化 | [LM-Steer](replications/lm_steer/) · [LoReFT](replications/loreft/) · [BiPO](replications/bipo/) |
| MoE | [SteerMoE](replications/steermoe/) |

## 引用

如果 EasySteer 对你的研究有帮助，请引用我们的论文：

```bibtex
@article{xu2025easysteer,
  title={EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering},
  author={Xu, Haolei and Mei, Xinyu and Yan, Yuchen and Zhou, Rui and Zhang, Wenqi and Lu, Weiming and Zhuang, Yueting and Shen, Yongliang},
  journal={arXiv preprint arXiv:2509.25175},
  year={2025}
}
```

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。

## 使用声明

LLM 干预技术具有双重用途。EasySteer 主要作为推进模型安全研究的工具而开发，而非用于绕过安全机制：干预应仅用于正当的研究与增强安全的应用；任何行为修改必须向最终用户明确披露；所有应用都必须遵守相关伦理准则与法律法规。

## 致谢

感谢 [vLLM](https://github.com/vllm-project/vllm) 项目提供的高性能推理框架，以及 [pyreft](https://github.com/stanfordnlp/pyreft) 等项目在表示学习领域的贡献。相关项目：[EasyEdit](https://github.com/zjunlp/EasyEdit) · [pyreft](https://github.com/stanfordnlp/pyreft) · [repeng](https://github.com/vgel/repeng) · [vLLM](https://github.com/vllm-project/vllm)

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
