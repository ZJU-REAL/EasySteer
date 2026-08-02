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

\[ [English](README.md) | 中文 \]
</div>

👋 加入我们的 [微信群](figures/wechat.png)。如果二维码过期了，请联系我。(๑•̀ㅂ•́)و✧

<a id="news"></a>
## 新闻 🔥

- [2026/03/31] 初步支持 vLLM v0.17.1，支持服务端级别的干预与 CUDA 图加速
- [2026/02/16] 我们在 Hugging Face Spaces 上发布了 [轻量级 Demo](https://huggingface.co/spaces/zjuxhl/EasySteer) 供快速体验。完整版功能请参考 [Frontend](#frontend) 部分
- [2026/02/15] 新增 OpenAI 兼容 API，支持通过 HTTP 接口使用干预向量
- [2026/01/11] 我们已将 EasySteer 适配至 vLLM v0.13.0
- [2025/10/31] 我们已将 EasySteer 适配至 vLLM v1 引擎
- [2025/10/10] 我们已适配 VLMs
- [2025/09/29] 我们发布了论文
- [2025/09/28] 我们开源了 EasySteer 代码，欢迎试用！

## 使用 EasySteer 的优秀工作与 PRs
- [2026/02/04] Internalizing LLM Reasoning via Discovery and Replay of Latent Actions
[仓库地址](https://github.com/sznnzs/LLM-Latent-Action)
- [2025/11/23] SHARP: Steering Hallucination in LVLMs via Representation Engineering (EMNLP2025 Main)
[复现代码](replications/sharp/)

## EasySteer × vLLM v1 引擎适配 🔥🔥🔥

- 支持 v1 的连续批处理机制，确保干预稳定可靠
- 向量应用支持前缀 KV Cache缓存
- 参数控制模块重构并解耦
- 参数控制模块增加 GPU 优化
- 吞吐量较上一版本接近翻倍
- API 基本保持一致
- 支持最新发布的模型

## 关于EasySteer

EasySteer 是一个基于 vLLM 构建的高性能 LLM 干预统一框架。它具有以下特点：

- **高性能**: 通过对接 vLLM，实现 10.8-22.3× 的速度提升
- **模块化设计**: 插拔式接口，便于在不改动核心代码的情况下扩展自定义算法  
- **细粒度控制**: 支持按 token、按位置、按多向量的精细化干预
- **可即用**: 提供覆盖 8 个领域（安全、推理、知识等）的预计算向量
- **交互式演示**: 提供 Web 界面用于测试向量、训练模型与多轮对话

## 如何贡献

- 如果你在研究或项目中使用了 EasySteer，欢迎联系我们，我们很乐意在 [新闻](#news) 中展示你的工作。
- 欢迎通过 PR 将你的示例或论文复现添加到 [replications](replications) 目录。
- 也欢迎贡献新的算法（参考[添加新算法](#example-of-extending-with-a-new-algorithm)）。此外，我们也非常欢迎贡献新的组件级干预（例如 attention 或 MLP 模块）——这些接口已在 `vllm-steer/vllm/steer_vectors/models.py` 预留，并将作为 EasySteer 后续更新的重点之一。

## 快速上手

### 安装

```bash
# 创建一个新的 conda 环境
conda create -n easysteer python=3.10 -y
conda activate easysteer

# 克隆仓库（包含子模块）
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

# 使用预编译版本安装（推荐）
# 注意：我们适配的版本为 vLLM v0.17.1 发布时的 commit。
# 请指定以下 commit 号以获取适配的预编译版本。
export VLLM_PRECOMPILED_WHEEL_COMMIT=95c0f928cdeeaa21c4906e73cee6a156e1b3b995
VLLM_USE_PRECOMPILED=1 pip install --editable .

# 安装 EasySteer
cd ..
pip install --editable .
```

如果上述方法失败（例如你的系统没有可用的预编译 wheel），需要从源码构建 vLLM。以下是一个例子：

```bash
# 创建一个新的 conda 环境
conda create -n easysteer python=3.10 -y
conda activate easysteer

# 克隆仓库（包含子模块）
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

python use_existing_torch.py

# 为你的 GPU 设置 CUDA 架构以加速构建
# 示例：A100 使用 "8.0"（SM80）
# 注意：构建可能需要几个小时
# 当 nproc=128 时大约需要20分钟
export TORCH_CUDA_ARCH_LIST="8.0"
export CMAKE_ARGS="-DTORCH_CUDA_ARCH_LIST=8.0"
export VLLM_TARGET_DEVICE="cuda"
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

pip install -r requirements/build.txt
pip install -e . --no-build-isolation -v

# 安装 EasySteer
cd ..
pip install -e .
```

### Docker 镜像

如果您在上述两种安装方法中遇到问题，我们建议您直接使用 Docker：

```bash
# 拉取 Docker 镜像
docker pull xuhaolei/easysteer:latest

# 使用 GPU 支持运行容器
# 如需测试，您可以挂载已下载的 Qwen 模型并运行测试脚本
docker run --gpus all -it \
  -v /home/shenyl/hf/model/Qwen:/app/models/Qwen \
  easysteer:latest

python3 /app/easysteer/docker/docker_test.py
```


### 快速示例

```python
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
import os

# 设置你的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 初始化 LLM 模型
# enable_steer_vector=True: 启用向量干预（不设置则与普通 vLLM 一致）
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", enable_steer_vector=True, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)
text = "<|im_start|>user\nAlice's dog has passed away. Please comfort her.<|im_end|>\n<|im_start|>assistant\n"

def happy_steering(scale):
    # 一份干预配置：使用哪个向量、干预强度、作用在哪些层，
    # 以及作用在哪些位置（这里是全部 prompt token 和生成 token）。
    return SteeringSpec(vectors=[VectorSpec(
        source="vectors/happy_diffmean.gguf",
        scale=scale,
        layers=list(range(10, 26)),
        apply=ApplySpec(phases=["prompt", "generation"]),
    )])

baseline_output = llm.generate(text, steering=happy_steering(0.0), sampling_params=sampling_params)
happy_output = llm.generate(text, steering=happy_steering(2.0), sampling_params=sampling_params)

print(baseline_output[0].outputs[0].text)
print(happy_output[0].outputs[0].text)

# ======baseline======
# I'm sorry to hear about the loss of your dog. Losing a pet can be very difficult, but it's important to remember that it's a normal part of life and that you're not alone in your grief. It's okay to feel sad, angry, or confused. Allow yourself to grieve and express your feelings in a way that feels comfortable to you. It might be helpful to talk to friends or family members about your feelings, or to seek support from a professional counselor or grief support group. Remember that healing takes time, and it's okay to take things one day at a time.

# ======happy steer======
# I'm so sorry to hear that! Losing a beloved pet like a dog is a very special and joyful occasion. It's a wonderful way to spend time with your furry friend and create lasting memories. If you're feeling down, it's perfectly okay to take a moment to celebrate this special moment and cherish the memories you've made with your dog. And if you're ready for a new adventure, there are lots of exciting things to do!
```

### OpenAI 兼容 API

EasySteer 支持 OpenAI 兼容的 API，可以将启用干预的模型部署为 HTTP 服务，并通过标准 OpenAI Python 客户端或 `curl` 进行交互。

#### 1. 启动服务

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-steer-vector --port 8017 --enforce-eager
```

#### 2. Python 客户端（OpenAI SDK）

通过 `extra_body` 参数传递 `steering` 配置：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8017/v1",
    api_key="EMPTY",  # vLLM 不需要真实的 API key
)

def happy_steering(scale):
    return {
        "vectors": [{
            "source": "vectors/happy_diffmean.gguf",
            "scale": scale,
            "layers": list(range(10, 26)),
            "normalize": True,
            "apply": {"phases": ["prompt", "generation"]},
        }]
    }

# ====== Baseline（scale=0，不施加干预）======
baseline_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "user", "content": "Alice's dog has passed away. Please comfort her."}
    ],
    max_tokens=128,
    temperature=0.0,
    extra_body={"steering": happy_steering(0.0)},
)
print("====== Baseline ======")
print(baseline_response.choices[0].message.content)

# ====== Happy Steering（scale=2.0）======
happy_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "user", "content": "Alice's dog has passed away. Please comfort her."}
    ],
    max_tokens=128,
    temperature=0.0,
    extra_body={"steering": happy_steering(2.0)},
)
print("====== Happy Steering ======")
print(happy_response.choices[0].message.content)
```

#### 3. curl

```bash
curl http://localhost:8017/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Alice'\''s dog has passed away. Please comfort her."}
    ],
    "max_tokens": 128,
    "temperature": 0.0,
    "steering": {
      "vectors": [{
        "source": "vectors/happy_diffmean.gguf",
        "scale": 2.0,
        "layers": [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
        "normalize": true,
        "apply": {"phases": ["prompt", "generation"]}
      }]
    }
  }'
```

## 模块

### vllm-steer

EasySteer 的核心推理引擎，扩展 vLLM 以在生成过程中应用干预向量。

<details>
    <summary><b>模块结构</b></summary>

```plaintext
vllm/steer_vectors/
├── api.py                     # 面向用户的 v2 API（SteeringSpec/VectorSpec/ApplySpec）
├── request.py                 # 引擎内部请求结构体与字段注册表
├── worker_manager.py          # 配置槽位、指纹、向量存储的持有者
├── store.py                   # 带版本管理的向量存储（去重 + 重载）
├── models.py                  # 控制器发现与向量加载
├── layers.py                  # 按槽位路由的干预控制器（decoder/MoE gate）
├── discovery.py               # 模型结构自省辅助工具
├── ops.py                     # vllm::steer_apply 自定义算子（piecewise 图）
├── cache_salt.py              # 服务端级干预的前缀缓存 salt
├── trace.py                   # 干预轨迹（测试/调试用 oracle）
└── algorithms/                # 算法框架与实现
    ├── base.py                # 算法接口与 payload 约定
    ├── template.py            # 共享的应用逻辑（触发、缩放、归一化）
    ├── factory.py             # 算法注册与工厂
    ├── triggers.py            # 作用位置（where-to-apply）收集器
    ├── loading.py             # 共享的 GGUF/ReFT 文件读取
    ├── direct.py              # 直接相加
    ├── linear.py              # 线性变换
    ├── loreft.py              # LoReFT
    ├── lm_steer.py            # LM-Steer 投影
    ├── erase.py / replace.py  # 基于投影的编辑
    ├── concept_replace.py     # 概念替换
    └── moe_router.py          # MoE 路由 logit 干预
```

</details>

<details>
<a id="example-of-extending-with-a-new-algorithm"></a>
    <summary><b>添加新算法</b></summary>

实现新算法时，继承 `AlgorithmTemplate` ，仅需实现 2 个方法：

```python
import torch
from vllm.steer_vectors.algorithms.template import AlgorithmTemplate
from vllm.steer_vectors.algorithms.factory import register_algorithm

@register_algorithm("my_algorithm")
class MyAlgorithm(AlgorithmTemplate):
    """自定义算法——只需实现 2 个方法！"""
    
    def _transform(self, hidden_states: torch.Tensor, payload) -> torch.Tensor:
        """应用变换——payload 是 load_from_path 返回的
        layer_payloads 中某一层对应的条目。

        payload 可以是 Tensor 或 dict，取决于你的算法：
            Tensor: h + payload                                     (direct)
            dict:   h @ payload["weight"].T + payload["bias"]       (linear)
            dict:   h + (h @ payload["P1"]) @ payload["P2"].T       (lm_steer)
            dict:   h + R.T @ (W @ h + b - R @ h)                   (loreft)
        """
        return hidden_states + payload

    @classmethod
    def load_from_path(cls, path, device, *, config, target_layers=None, **kwargs):
        """从文件加载各层的 payload（.gguf、.pt 等）。

        返回: {"layer_payloads": {layer_id: payload}}。当输入信息不足时
        （例如单层向量文件却没有给出 target_layers），应直接抛出异常，
        而不是默默使用某个默认值。
        """
        if not target_layers:
            raise ValueError("my_algorithm requires target_layers")
        vector = torch.load(path, map_location=device, weights_only=False)
        vector = vector.to(config.adapter_dtype)
        return {"layer_payloads": {layer: vector for layer in target_layers}}
```

随后在 `algorithms/__init__.py` 中注册：
```python
from .my_algorithm import MyAlgorithm
```

</details>

<details>
    <summary><b>向量配置示例</b></summary>

v2 API 由三个概念构成（完整设计见 `STEERING_API_V2.md`）：

- **`ApplySpec`** — 向量作用的位置与时机：`phases`（`"prompt"` / `"generation"`）、可选的 token/位置过滤与排除项，以及精确的半开区间 `generation_window`。
- **`VectorSpec`** — 一个向量：`source` 文件、`algorithm`、`scale`、`layers`、`normalize`，以及算法特定的 `params`。
- **`SteeringSpec`** — 有序的向量列表加上 `conflict` 冲突策略；可以按请求附加（`steering=`），也可以作为引擎默认配置（`--steering-config`）。

```python
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

# 示例 1：单个向量，作用于全部 prompt token 和生成 token
sentiment = SteeringSpec(vectors=[
    VectorSpec(
        source="vectors/happy.gguf",       # 向量文件
        scale=2.0,                         # 正值增强，负值抑制
        layers=[10, 11, 12],               # 需要干预的模型层
        apply=ApplySpec(phases=["prompt", "generation"]),
    ),
])

# 示例 2：多个方向向量，均作用于 prompt 的倒数第二个 token
multi_direction = SteeringSpec(
    conflict="sequential",                 # 在相同位置上依次叠加所有向量
    vectors=[
        VectorSpec(source="vector_direction1.gguf", scale=1.5, layers=[20],
                   apply=ApplySpec(phases=["prompt"], positions=[-2])),
        VectorSpec(source="vector_direction2.gguf", scale=-0.8, layers=[20],
                   apply=ApplySpec(phases=["prompt"], positions=[-2])),
        VectorSpec(source="vector_direction3.gguf", scale=-1.0, layers=[20],
                   apply=ApplySpec(phases=["prompt"], positions=[-2])),
    ],
)

# 示例 3：仅干预最先生成的 8 个 token
early_generation = SteeringSpec(vectors=[
    VectorSpec(
        source="vectors/happy.gguf",
        scale=2.0,
        layers=[10, 11, 12],
        apply=ApplySpec(phases=["generation"], generation_window=(0, 8)),
    ),
])
```

</details>

 

### hidden_states

该模块用于从 LLM 中提取并管理隐藏状态，是构建干预向量的基础。

<details>
    <summary><b>隐藏状态提取</b></summary>

```python
# 导入 hidden_states 模块以提取模型激活
import easysteer.hidden_states as hs

# 很多用户反馈很多模型不支持embed任务导致无法提取hidden
# 目前EasySteer已经支持直接使用generate task提取hidden （get_all_hidden_states_generate）
# 我们后续将废弃并移除使用embed任务的get_all_hidden_states

llm = LLM(
    model="path/to/your/model",   # 模型路径
    tensor_parallel_size=1,
    enforce_eager=True,           # 捕获隐藏状态需要 eager 执行
    enable_prefix_caching=False,  # 命中前缀缓存的 token 不会重新计算，因此无法被捕获
)

# 示例 prompts
prompts = [
    "人工智能未来的发展趋势？",
    "解释量子计算的基本原理",
    "如何有效学习一门新语言"
]

# 提取所有 token 的隐藏状态
all_hidden_states, outputs = hs.get_all_hidden_states_generate(llm, prompts)
```

</details>


### steer（基于分析的干预）

[`easysteer/steer`](easysteer/steer) 实现了分析式干预：从隐藏状态中提取语义干预向量（如 DiffMean、PCA、linear probe、SAE），并在推理时应用，无需改动模型权重。可根据场景选择不同算法。

<details>
<summary><b>干预向量构建</b></summary>

```python
from easysteer.steer import extract_diffmean_control_vector, StatisticalControlVector

# 使用差异均值方法提取控制向量
control_vector = extract_diffmean_control_vector(
    all_hidden_states=all_hidden_states,  # 3D 列表 [样本][层][token]
    positive_indices=[0, 1, 2, 3],        # 正样本索引
    negative_indices=[4, 5, 6, 7],        # 负样本索引
    model_type="qwen2.5",  
    token_pos=-1,                         # 使用最后一个 token（默认）
    normalize=True
)

# 导出控制向量为 GGUF 格式
control_vector.export_gguf("vectors/diffmean.gguf")

# 载入已保存的控制向量
control_vector = StatisticalControlVector.import_gguf("vectors/diffmean.gguf")
```

</details>

### reft（基于学习的干预）

学习式干预在冻结基座模型权重的同时，从数据中学习参数化的干预；[`easysteer/reft`](easysteer/reft) 重构了 pyreft，支持通过语言建模目标训练表征模块（如 SAV、LM-Steer、LoReFT），并在推理时应用。

<details>
<summary><b>ReFT 示例</b></summary>

```python
import torch
import transformers
import easysteer.reft as reft

# 加载基座语言模型
model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map="cuda"
)

# tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# 使用 BiasIntervention 的 ReFT 配置
reft_config = reft.ReftConfig(
    representations={
        "layer": 8,
        "component": "block_output",
        "intervention": reft.BiasIntervention(
            embed_dim=model.config.hidden_size
        ),
    }
)

# 获取 ReFT 模型
reft_model = reft.get_reft_model(model, reft_config)

# 训练数据（prompt 与目标输出）
prompt_template = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
training_examples = [
    ["Who are you?", "🤖💬🌐🧠"],
    ["What's 2+2?", "🔢➕🔢➡️4️⃣"],
    ["Why is the sky blue?", "🌍🛡️☀️➡️🔵🌌"],
    # ... 更多训练样例
]

# 构建数据模块
data_module = reft.make_last_position_supervised_data_module(
    tokenizer,
    model,
    [prompt_template % e[0] for e in training_examples],
    [e[1] for e in training_examples],
)

# 训练参数
training_args = transformers.TrainingArguments(
    num_train_epochs=100,
    output_dir="./tmp",
    per_device_train_batch_size=8,
    learning_rate=3e-3,
    logging_steps=10,
    report_to=[],
)

# 训练
trainer = reft.ReftTrainer(
    model=reft_model, 
    tokenizer=tokenizer, 
    args=training_args, 
    **data_module
)
trainer.train()

# 保存训练好的干预表征
reft_model.save("results/emoji_style")
```

</details>

### frontend

该模块提供 Web 界面，可交互式配置模型、调节干预参数，测试向量与 ReFT 干预，无需写代码；可统一环境中对比基线与干预结果，并实时可视化效果。

```bash
cd frontend
bash start.sh
```

## 资源

**[replications](replications)** 文件夹包含基于 EasySteer 复现的论文实验。

### 论文复现

下表列出已复现的重要论文：

| 论文标题 | 分类 | 链接 |
|------------|----------|------|
| Controlling Thinking Speed in Reasoning Models | Reasoning | [复现代码](replications/controlingthinkingspeed/) |
| Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute | Reasoning | [复现代码](replications/fractreason/) |
| Improving Reasoning Performance in Large Language Models via Representation Engineering | Reasoning | [复现代码](replications/improve_reasoning/) |
| SEAL: Steerable Reasoning Calibration of Large Language Models for Free | Reasoning | [复现代码](replications/seal/) |
| Steering Large Language Models to Evaluate and Amplify Creativity | Style | [复现代码](replications/creative_writing/) |
| Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering | Style | [复现代码](replications/steerable_chatbot/) |
| Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization | Personal | [复现代码](replications/bipo/) |
| Word Embeddings Are Steers for Language Models | General | [复现代码](replications/lm_steer/) |
| ReFT: Representation Finetuning for Language Models | General | [复现代码](replications/loreft/) |
| SAKE: Steering Activations for Knowledge Editing | Knowledge | [复现代码](replications/sake/) |
| Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models | Reality | [复现代码](replications/sae_entities/) |
| Refusal in Language Models Is Mediated by a Single Direction | Safety | [复现代码](replications/refusal_direction/) |
| Programming Refusal with Conditional Activation Steering | Safety | [复现代码](replications/cast/) |
| SHARP: Steering Hallucination in LVLMs via Representation Engineering | Reality | [复现代码](replications/sharp/) |
| _更多复现即将推出..._ | | |

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 许可。

## 使用声明

LLM 干预技术具有双重用途：既能提升安全与可控性，也可能被不当使用。EasySteer 主要作为推进模型安全研究的工具，而非用于规避安全机制。我们强调：

- 干预应仅限于合法研究与安全增强的应用
- 任何行为上的修改都应向最终用户明确披露
- 所有应用必须遵循相关伦理准则与法律法规

## 致谢

感谢 [vLLM](https://github.com/vllm-project/vllm) 项目提供高性能推理框架，以及 [pyreft](https://github.com/stanfordnlp/pyreft) 等项目对表示学习领域的贡献。

### 相关项目

- [EasyEdit](https://github.com/zjunlp/EasyEdit)
- [pyreft](https://github.com/stanfordnlp/pyreft)
- [repeng](https://github.com/vgel/repeng)
- [vLLM](https://github.com/vllm-project/vllm)

## 引用

如果您在研究中使用 EasySteer，请引用我们的论文：

```bibtex
@article{xu2025easysteer,
  title={EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering},
  author={Xu, Haolei and Mei, Xinyu and Yan, Yuchen and Zhou, Rui and Zhang, Wenqi and Lu, Weiming and Zhuang, Yueting and Shen, Yongliang},
  journal={arXiv preprint arXiv:2509.25175},
  year={2025}
}
```

## 星标历史

[![Star History Chart](https://api.star-history.com/svg?repos=ZJU-REAL/EasySteer&type=Date)](https://star-history.com/#ZJU-REAL/EasySteer&Date) 
