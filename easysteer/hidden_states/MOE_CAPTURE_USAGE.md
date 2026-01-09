# MoE Router Logits Capture - 使用指南

## 概述

MoE Router Logits提取功能已集成到EasySteer的统一捕获系统中，与Hidden States提取共享相同的架构。

## 架构说明

### 统一的Capture Mixin

所有提取功能（Hidden States、MoE Router Logits等）现在都统一管理在：
- **vLLM内部**: `CaptureModelRunnerMixin` (`vllm/v1/worker/capture_model_runner_mixin.py`)
- **外部API**: `easysteer.hidden_states` 模块

### 设计优势

1. **集中管理**: 所有提取功能在一个mixin中
2. **易于扩展**: 未来可以轻松添加新的提取功能（attention weights, gradients等）
3. **统一接口**: 类似的API设计，易于学习和使用
4. **向后兼容**: 旧代码继续工作

## 快速开始

### 方式1：使用`task="embed"`（推荐用于大多数MoE模型）

```python
import easysteer.hidden_states as hs
from vllm import LLM
from vllm.hidden_states import print_moe_router_logits_summary

# 1. 加载MoE模型（如Mixtral）
llm = LLM(model="mistralai/Mixtral-8x7B-v0.1", task="embed")

# 2. 提取router logits
router_logits, outputs = hs.get_moe_router_logits(
    llm, 
    ["The capital of France is Paris."]
)

# 3. 查看结果 - 现在返回的是真实的torch.Tensor
print_moe_router_logits_summary(router_logits)
# 输出示例:
# 📊 Captured 32 MoE layers:
#   Layer  0: 8 tokens × 8 experts, dtype torch.bfloat16, device cpu
#   Layer  1: 8 tokens × 8 experts, dtype torch.bfloat16, device cpu
#   ...
```

### 方式2：使用`generate`任务（适用于VLM等特殊模型）

对于像Qwen3-VL这样的视觉语言模型，使用generate任务：

```python
import easysteer.hidden_states as hs
from vllm import LLM
from vllm.hidden_states import print_moe_router_logits_summary

# 1. 加载模型（不指定task，默认为generate）
llm = LLM(
    model="Qwen/Qwen3-VL-30B-A3B-Thinking",
    tensor_parallel_size=4,
    trust_remote_code=True,
)

# 2. 使用generate模式提取
router_logits, outputs = hs.get_moe_router_logits_generate(
    llm, 
    prompts=["What is AI?"],
    max_tokens=10
)

# 3. 查看结果
print_moe_router_logits_summary(router_logits)
```

### 分析专家选择

```python
# 分析专家使用模式
analysis = hs.analyze_expert_usage(router_logits, top_k=2)

# 查看每层的专家使用统计
for layer_id in router_logits.keys():
    print(f"\nLayer {layer_id}:")
    print(f"  专家使用次数: {analysis['expert_usage_counts'][layer_id]}")
    print(f"  负载均衡指标: {analysis['load_balance'][layer_id]:.3f}")
```

### 手动分析router logits

```python
import torch

# 获取特定层的logits
layer_10_logits = router_logits[10]  # Shape: (num_tokens, 8) for Mixtral

# 计算专家选择概率
probs = torch.softmax(layer_10_logits, dim=-1)

# 获取top-k专家
top2_probs, top2_ids = torch.topk(probs, k=2, dim=-1)

print(f"Token 0选择的专家: {top2_ids[0]}")  # e.g., tensor([3, 5])
print(f"对应的权重: {top2_probs[0]}")      # e.g., tensor([0.6, 0.4])
```

## 与Hidden States同时使用

```python
# 可以同时捕获hidden states和router logits
hidden_states, _ = hs.get_all_hidden_states(llm, texts)
router_logits, _ = hs.get_moe_router_logits(llm, texts)

# 分析专家选择与hidden states的关系
for layer_id in router_logits.keys():
    layer_hidden = hidden_states[0][layer_id]  # 第一个样本的hidden states
    logits = router_logits[layer_id]
    
    # 分析：哪些hidden states模式对应哪些专家选择
    # ...
```

## 高级用法

### 类接口

```python
# 使用类接口进行更细粒度的控制
capture = hs.MoERouterLogitsCaptureV1()
router_logits, outputs = capture.get_router_logits(llm, texts)
```

### 支持的MoE模型

自动识别以下MoE架构：
- Mixtral (`MixtralMoE`)
- DeepSeek-V2 (`DeepseekV2MoE`)
- Qwen-MoE (`QwenMoE`, `Qwen2MoE`)
- DBRX (`DbrxExperts`)
- Arctic (`ArcticMoE`)
- Ernie 4.5 MoE (`Ernie4MoE`)
- GLM-4-MoE (`GLMMoE`)

## 应用场景

### 1. 专家专业化分析

```python
# 分析不同专家处理不同类型输入的模式
math_texts = ["What is 2+2?", "Solve x^2 = 4"]
code_texts = ["def hello():", "class MyClass:"]

math_logits, _ = hs.get_moe_router_logits(llm, math_texts)
code_logits, _ = hs.get_moe_router_logits(llm, code_texts)

# 比较专家选择差异
# ...
```

### 2. 负载均衡优化

```python
# 识别未充分利用的专家
analysis = hs.analyze_expert_usage(router_logits)

for layer_id, usage in analysis['expert_usage_counts'].items():
    underused = [i for i, count in enumerate(usage) if count < mean_usage * 0.5]
    print(f"Layer {layer_id} underused experts: {underused}")
```

### 3. 模型行为理解

```python
# 分析routing entropy - 理解模型的专家选择多样性
for layer_id, logits in router_logits.items():
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    print(f"Layer {layer_id} routing entropy: {entropy:.3f}")
```

## 技术细节

### 捕获机制

1. **包装器模式**: 在模型加载时包装MoE层
2. **Gate拦截**: 捕获gate模块的输出（router_logits）
3. **多批次支持**: 自动合并多次forward的结果
4. **CPU移动**: 自动移动到CPU以节省GPU内存

### 输出格式

- **router_logits**: `Dict[int, torch.Tensor]`
  - Key: Layer ID (0-based)
  - Value: Tensor of shape `(num_tokens, n_experts)`
  - 未归一化的logits（使用softmax获得概率）

### 性能考虑

- **内存**: Router logits通常很小 (num_tokens × n_experts)
- **速度**: 捕获开销<1%（仅复制gate输出）
- **GPU内存**: 自动移至CPU，不占用GPU内存

## 未来扩展

统一的Capture架构支持未来添加：
- Attention weights capture
- Gradient capture
- Specific layer activation capture
- Expert output capture (individual expert outputs)

添加新功能只需在`CaptureModelRunnerMixin`中添加相应方法。

## 故障排除

### Q: 模型没有MoE层，会报错吗？
A: 不会。系统会检测到没有MoE层并返回空字典，同时给出警告。

### Q: 可以只捕获特定层的router logits吗？
A: 当前版本捕获所有MoE层。未来可以添加layer过滤功能。

### Q: router_logits和topk选择的关系？
A: router_logits是原始分数，需要softmax+topk获得实际选择的专家。

## 参考

- [Hidden States Capture文档](./README.md)
- [vLLM V1架构](../../vllm-steer/vLLM_V1_Adaptation_Plan.md)
- [Mixtral论文](https://arxiv.org/abs/2401.04088)

