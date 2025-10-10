<div align="center">
<h3>
    <img src="figures/logo.png" width="50%"><br>
    A Unified Framework for High-Performance and Extensible LLM Steering
</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-REAL/EasySteer?style=social)](https://github.com/ZJU-REAL/EasySteer/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/commits/main)
[![GitHub](https://img.shields.io/github/license/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2509.25175-b31b1b.svg)](https://arxiv.org/abs/2509.25175)

\[ [English](README.md) | 中文 \]
</div>

👋 加入我们的 [微信群](figures/wechat.png).

## 新闻 🔥

- [2025/09/29] 论文已发布。
- [2025/09/28] 开源 EasySteer 代码 —— 欢迎试用！

## 关于

基于 vLLM 构建，EasySteer 是一个高性能 LLM 干预的统一框架。它具有以下特点：

- **高性能**: 通过对接 vLLM，实现 5.5-11.4× 的速度提升
- **模块化设计**: 插拔式接口，便于在不改动核心代码的情况下扩展自定义算法  
- **细粒度控制**: 支持按 token、按位置、按多向量的精细化干预
- **可即用**: 提供覆盖 8 个领域（安全、推理、知识等）的预计算向量
- **交互式演示**: 提供 Web 界面用于测试向量、训练模型与多轮对话

## 如何贡献

- 我们非常欢迎通过 PR 的方式进行贡献。
- 如果您有与 LLM steering 相关的工作，我们很期待您将复现结果添加到 `replications/` 文件夹中。
  - 理想情况下，请附带一个简易的向量提取脚本或预计算好的向量（例如 GGUF），并提供一个用于推理和对比的简易 steer 脚本。
- 如果您希望在 EasySteer 中集成新的算法，请参考 “新算法扩展示例” 部分的说明。

## 快速上手

### 安装

#### 针对 x86_64 架构

```bash
# 创建一个新的 conda 环境
conda create -n easysteer python=3.10 -y
conda activate easysteer

# 克隆仓库（包含子模块）
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

# 使用预编译版本安装（推荐）
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/cede942b87b5d8baa0b95447f3e87e3c600ff5f5/vllm-0.9.2rc2.dev34%2Bgcede942b8-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .
pip install transformers==4.53.1

# 安装 EasySteer
cd ..
pip install --editable .
```

#### 针对 ARM (aarch64) 架构

需要从源码构建 vLLM（因为预编译只支持 x86_64）。

```bash
# 创建一个新的 conda 环境
conda create -n easysteer python=3.10 -y
conda activate easysteer

# 克隆仓库（包含子模块）
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

python use_existing_torch.py

# 为你的 GPU 设置 CUDA 架构以加速构建
# 例如：A100 使用 "8.0"（SM80）
# 注意：这一过程可能耗时数小时
export TORCH_CUDA_ARCH_LIST="8.0"
export CMAKE_ARGS="-DTORCH_CUDA_ARCH_LIST=8.0"
export VLLM_TARGET_DEVICE="cuda"

pip install -r requirements/build.txt
pip install -e . --no-build-isolation -v

# 安装 EasySteer
cd ..
pip install -e .
pip install transformers==4.53.1
```

### 快速示例

```python
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest
import os

# 由于当前干预功能暂不支持 v1，需设置使用 vLLM v0
os.environ["VLLM_USE_V1"]="0"

# 设置你的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 初始化 LLM 模型
# enable_steer_vector=True: 启用向量干预（不设置则与普通 vLLM 一致）
# enforce_eager=True: 确保干预时的可靠性与稳定性（强烈建议）
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", enable_steer_vector=True, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)
text = "<|im_start|>user\nAlice's dog has passed away. Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
target_layers = list(range(10,26))

baseline_request = SteerVectorRequest("baseline", 1, steer_vector_local_path="vectors/happy_diffmean.gguf", scale=0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
baseline_output = llm.generate(text, steer_vector_request=baseline_request, sampling_params=sampling_params)

happy_request = SteerVectorRequest("happy", 2, steer_vector_local_path="vectors/happy_diffmean.gguf", scale=2.0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
happy_output = llm.generate(text, steer_vector_request=happy_request, sampling_params=sampling_params)

print(baseline_output[0].outputs[0].text)
print(happy_output[0].outputs[0].text)

# ======baseline======
# I'm sorry to hear about the loss of your dog. Losing a pet can be very difficult, but it's important to remember that it's a normal part of life and that you're not alone in your grief. It's okay to feel sad, angry, or confused. Allow yourself to grieve and express your feelings in a way that feels comfortable to you. It might be helpful to talk to friends or family members about your feelings, or to seek support from a professional counselor or grief support group. Remember that healing takes time, and it's okay to take things one day at a time.

# ======happy steer======
# I'm so sorry to hear that! Losing a beloved pet like a dog is a very special and joyful occasion. It's a wonderful way to spend time with your furry friend and create lasting memories. If you're feeling down, it's perfectly okay to take a moment to celebrate this special moment and cherish the memories you've made with your dog. And if you're ready for a new adventure, there are plenty of exciting things to do!
```

## 模块

### vllm-steer

EasySteer 的核心推理引擎，扩展 vLLM 以在生成过程中应用干预向量。

<details>
    <summary><b>内部结构</b></summary>

`vllm-steer` 的核心功能位于 `vllm/steer_vectors` 目录，文件结构如下：

```plaintext
vllm/steer_vectors/
├── __init__.py                # 模块入口
├── request.py                 # 请求与配置定义
├── models.py                  # 模型集成与向量注册
├── layers.py                  # 自定义层实现
├── worker_manager.py          # 工作线程管理
└── algorithms/                # 各类干预算法实现
    ├── __init__.py            # 算法注册
    ├── base.py                # 算法基类与接口定义
    ├── factory.py             # 算法工厂（创建算法实例）
    ├── direct.py              # 直接干预算法
    ├── loreft.py              # LoReFT 算法实现
    ├── xxx.py                 # 其他算法
    ├── multi_vector.py        # 多向量组合算法
    └── template.py            # 新算法模板示例
```

</details>

<details>
    <summary><b>核心组件</b></summary>

1. **请求与配置系统**（`request.py`）:
   - `SteerVectorRequest`: 定义干预向量请求格式，支持单向量与多向量模式
   - `VectorConfig`: 多向量模式下的单向量配置定义

2. **算法框架**（`algorithms/base.py`）:
   - `BaseSteerVectorAlgorithm`: 所有干预算法的抽象基类，定义标准接口
   - 提供位置解析、触发条件检查等通用功能

3. **算法工厂**（`algorithms/factory.py`）:
   - 根据配置动态创建合适的算法实例
   - 支持算法注册机制，便于扩展

4. **向量应用实现**:
   - `direct.py`: 直接加性干预
   - `loreft.py`: LoReFT 低秩适配干预方法
   - `multi_vector.py`: 多向量组合策略

</details>

<details>
    <summary><b>扩展机制</b></summary>

`vllm-steer` 提供灵活扩展机制，便于研究者实现并集成自定义干预算法：

1. **基于接口的插件架构**:
   - 所有算法继承自 `BaseSteerVectorAlgorithm`
   - 实现标准接口方法即可新增算法，无需修改核心代码

2. **算法注册系统**:
   - 在 `algorithms/__init__.py` 中注册新算法
   - 工厂模式自动加载并实例化算法

3. **模板示例**:
   - `template.py` 提供开发模板与注释说明
   - 按模板实现可与框架无缝集成

4. **多层级干预点**:
   - 支持在不同模型层级（注意力、FFN 等）应用干预
   - 通过 `forward_decoder_layer`、`forward_mlp_layer` 等钩子实现

</details>

<details>
    <summary><b>新算法扩展示例</b></summary>

要添加新的干预算法，只需：

1. 创建继承 `BaseSteerVectorAlgorithm` 的新类
2. 实现必要的接口方法（如 `load_from_path`、`apply_intervention` 等）
3. 在算法注册系统中登记
4. 通过配置使用新算法

```python
# 示例：实现一个新的干预算法
from vllm.steer_vectors.algorithms.base import BaseSteerVectorAlgorithm
import torch

class MyCustomAlgorithm(BaseSteerVectorAlgorithm):
    """自定义干预算法实现"""
    
    @classmethod
    def load_from_path(cls, path, device, **kwargs):
        # 向量文件加载实现
        vector_data = torch.load(path, map_location=device)
        return {"vector": vector_data, "other_params": ...}
    
    def __init__(self, layer_id=None):
        super().__init__(layer_id)
        self.vector = None
        self.scale = 1.0
        
    def set_steer_vector(self, index, vector, scale=1.0, **kwargs):
        self.vector = vector
        self.scale = scale
    
    def apply_intervention(self, hidden_states):
        # 自定义干预逻辑
        if self.vector is not None:
            return hidden_states + self.scale * self.vector
        return hidden_states
    
    # 实现其他必要接口方法...

# 在 algorithms/__init__.py 中注册：
# ALGORITHM_CLASSES["my_custom"] = MyCustomAlgorithm
```

通过模块化设计，研究者可聚焦于干预算法的核心逻辑，而无需深入底层推理引擎细节。

</details>

<details>
    <summary><b>向量配置示例</b></summary>

```python
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig

# 示例 1：单向量干预配置
single_vector_request = SteerVectorRequest(
    steer_vector_name="sentiment_control",        # 向量名称（用于日志与调试）
    steer_vector_id=1,                            # 向量 ID（内部标识）
    steer_vector_local_path="vectors/happy.gguf", # 向量文件路径
    scale=2.0,                                    # 应用强度（正增强，负抑制）
    target_layers=[10, 11, 12],                   # 目标层（指定作用的模型层）
    prefill_trigger_tokens=[-1],                  # 预填阶段干预 token（-1 表示全部）
    generate_trigger_tokens=[-1]                  # 生成阶段干预 token（-1 表示全部）
)

# 示例 2：多向量干预配置
multi_vector_request = SteerVectorRequest(
    # 向量请求基本信息
    steer_vector_name="multi_direction_control",   # 组合向量名称
    steer_vector_id=2,                             # 组合向量 ID
    
    # 多个方向的干预向量
    vector_configs=[
        # 第一个向量配置
        VectorConfig(
            path="vector_direction1.gguf",          # 向量文件路径
            scale=1.5,                              # 正向强度（增强）
            target_layers=[20],                     # 作用于第 20 层
            prefill_trigger_positions=[-2],         # 干预 prompt 倒数第 2 个位置
            algorithm="direct",                     # 应用算法
            normalize=False                         # 是否归一化
        ),
        
        # 第二个向量配置
        VectorConfig(
            path="vector_direction2.gguf",          # 向量文件路径
            scale=-0.8,                             # 负向强度（抑制）
            target_layers=[20],                     # 作用于第 20 层
            prefill_trigger_positions=[-2],         # 干预 prompt 倒数第 2 个位置
            algorithm="direct",                     # 应用算法
            normalize=False                         # 是否归一化
        ),
        
        # 第三个向量配置
        VectorConfig(
            path="vector_direction3.gguf",          # 向量文件路径
            scale=-1.0,                             # 负向强度（抑制）
            target_layers=[20],                     # 作用于第 20 层
            prefill_trigger_positions=[-2],         # 干预 prompt 倒数第 2 个位置
            algorithm="direct",                     # 应用算法
            normalize=False                         # 是否归一化
        ),
    ],
    
    # 多向量干预附加参数
    debug=False,                                    # 是否输出调试信息
    conflict_resolution="sequential"                # 冲突处理策略：顺序应用
)
```

</details>

### hidden_states

该模块用于从 LLM 中提取并管理隐藏状态，是构建干预向量的基础。

<details>
    <summary><b>隐藏状态提取</b></summary>

```python
# 导入 hidden_states 模块以提取模型激活
import easysteer.hidden_states as hs

# 以 reward 模式创建 LLM 实例
# 注意：这允许我们提取隐藏状态而非生成文本
llm = LLM(
    model="path/to/your/model",  # 模型路径
    task="reward",               # 使用 reward 任务获取隐藏状态
    tensor_parallel_size=1
)

# 示例 prompts
prompts = [
    "人工智能未来的发展趋势？",
    "解释量子计算的基本原理",
    "如何有效学习一门新语言"
]

# 提取所有 token 的隐藏状态
all_hidden_states, outputs = hs.get_all_hidden_states(llm, prompts)
```

</details>


### steer（基于分析的干预）

`easysteer/steer` 实现了分析式干预：从隐藏状态中提取语义干预向量（如 DiffMean、PCA、linear probe、SAE），并在推理时应用，无需改动模型权重。可根据场景选择不同算法。

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

学习式干预在冻结基座模型权重的同时，从数据中学习参数化的干预；`easysteer/reft` 重实现了 pyreft，支持通过语言建模或偏好目标训练表征模块（如 SAV、LM-Steer、LoReFT），并在推理时应用。

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

**`replications`** 文件夹包含基于 EasySteer 复现的论文实验。

### 论文复现

下表列出已复现的重要论文：

| 论文标题 | 分类 | 链接 |
|------------|----------|------|
| Activation Steering for Chain-of-Thought Compression | Reasoning | [复现代码](replications/asc/) |
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