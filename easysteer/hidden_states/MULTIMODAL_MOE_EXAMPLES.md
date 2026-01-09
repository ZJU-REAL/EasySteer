# MoE Router Logits捕获：多模态支持

`get_moe_router_logits_generate`现在支持多模态输入，适用于视觉语言模型（VLM）如Qwen3-VL。

## 快速开始

### 1. 纯文本输入（向后兼容）

```python
import easysteer.hidden_states as hs
from vllm import LLM

# 初始化模型
llm = LLM(
    model="Qwen/Qwen3-VL-30B-A3B-Thinking",
    tensor_parallel_size=4,
    trust_remote_code=True
)

# 纯文本输入
router_logits, outputs = hs.get_moe_router_logits_generate(
    llm,
    prompts=["What is artificial intelligence?"],
    max_tokens=10
)
```

### 2. 多模态输入（Text + Image）

```python
# 多模态输入格式（根据vLLM的API）
multimodal_prompts = [
    {
        "prompt": "Describe what you see in this image.",
        "multi_modal_data": {
            "image": "https://example.com/cat.jpg"
        }
    }
]

# 捕获router logits
router_logits, outputs = hs.get_moe_router_logits_generate(
    llm,
    prompts=multimodal_prompts,
    max_tokens=20
)

# 分析结果
print(f"捕获了 {len(router_logits)} 个MoE层")
for layer_id, logits in router_logits.items():
    print(f"Layer {layer_id}: {logits.shape}")
```

## 支持的输入格式

### 格式1: 纯文本 (List[str])

```python
prompts = [
    "Hello, how are you?",
    "What is the weather like?"
]

router_logits, outputs = hs.get_moe_router_logits_generate(
    llm, prompts=prompts, max_tokens=5
)
```

### 格式2: 多模态字典 (List[Dict])

```python
from PIL import Image

# 本地图片文件
prompts = [
    {
        "prompt": "What is in this image?",
        "multi_modal_data": {
            "image": Image.open("path/to/image.jpg")
        }
    }
]

# 或者使用URL
prompts = [
    {
        "prompt": "Describe this scene.",
        "multi_modal_data": {
            "image": "https://example.com/scene.jpg"
        }
    }
]

router_logits, outputs = hs.get_moe_router_logits_generate(
    llm, prompts=prompts, max_tokens=30
)
```

### 格式3: 批量多模态输入

```python
# 多个样本，每个包含图片
prompts = [
    {
        "prompt": "What animal is this?",
        "multi_modal_data": {"image": "cat.jpg"}
    },
    {
        "prompt": "What color is the sky?",
        "multi_modal_data": {"image": "sky.jpg"}
    },
    {
        "prompt": "How many people are in the photo?",
        "multi_modal_data": {"image": "crowd.jpg"}
    }
]

# 按样本划分捕获
router_logits, outputs = hs.get_moe_router_logits_generate(
    llm,
    prompts=prompts,
    max_tokens=10,
    split_by_samples=True  # 每个样本单独返回
)

# router_logits[0][layer_id] = 第1个样本的router logits
# router_logits[1][layer_id] = 第2个样本的router logits
```

## 分析多模态输入的专家使用

### 示例1: 对比纯文本 vs 图文输入的专家使用

```python
import easysteer.hidden_states as hs

# 1. 纯文本输入
text_only = ["Describe a cat."]
text_logits, _ = hs.get_moe_router_logits_generate(
    llm, prompts=text_only, max_tokens=20
)

# 2. 图文输入
image_text = [
    {
        "prompt": "Describe this cat.",
        "multi_modal_data": {"image": "cat.jpg"}
    }
]
image_logits, _ = hs.get_moe_router_logits_generate(
    llm, prompts=image_text, max_tokens=20
)

# 3. 分析差异
text_analysis = hs.analyze_expert_usage(text_logits, top_k=2, normalize=True)
image_analysis = hs.analyze_expert_usage(image_logits, top_k=2, normalize=True)

# 对比第10层
print("Layer 10 expert usage comparison:")
print(f"{'Expert':<10} {'Text Only':<12} {'Text+Image':<12} {'Diff':<12}")
for expert_id in range(8):
    text_freq = text_analysis['expert_usage_counts'][10][expert_id]
    image_freq = image_analysis['expert_usage_counts'][10][expert_id]
    diff = image_freq - text_freq
    print(f"{expert_id:<10} {text_freq:<12.3f} {image_freq:<12.3f} {diff:+.3f}")
```

### 示例2: 分析图像理解的不同阶段

假设我们知道prompt有10个tokens，生成了20个tokens：

```python
# 捕获多模态输入的logits
prompts = [{
    "prompt": "Describe this image in detail.",
    "multi_modal_data": {"image": "complex_scene.jpg"}
}]

router_logits, outputs = hs.get_moe_router_logits_generate(
    llm, prompts=prompts, max_tokens=20
)

# 分析不同阶段
prompt_length = 10

# 阶段1: Prompt处理（包括图像编码）
stage1 = hs.analyze_expert_usage(
    router_logits,
    top_k=2,
    token_range=(0, prompt_length),
    normalize=True
)

# 阶段2: 生成前半部分
stage2 = hs.analyze_expert_usage(
    router_logits,
    top_k=2,
    token_range=(prompt_length, prompt_length + 10),
    normalize=True
)

# 阶段3: 生成后半部分
stage3 = hs.analyze_expert_usage(
    router_logits,
    top_k=2,
    token_range=(prompt_length + 10, None),
    normalize=True
)

# 可视化各阶段专家使用
import matplotlib.pyplot as plt
import numpy as np

layer_id = 10
experts = range(8)
stage1_freq = [stage1['expert_usage_counts'][layer_id][i] for i in experts]
stage2_freq = [stage2['expert_usage_counts'][layer_id][i] for i in experts]
stage3_freq = [stage3['expert_usage_counts'][layer_id][i] for i in experts]

x = np.arange(len(experts))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, stage1_freq, width, label='Prompt (w/ Image)')
ax.bar(x, stage2_freq, width, label='Generation Early')
ax.bar(x + width, stage3_freq, width, label='Generation Late')

ax.set_xlabel('Expert ID')
ax.set_ylabel('Activation Frequency')
ax.set_title(f'Expert Usage Across Generation Stages (Layer {layer_id})')
ax.set_xticks(x)
ax.set_xticklabels(experts)
ax.legend()
plt.tight_layout()
plt.savefig('multimodal_expert_stages.png')
```

### 示例3: 不同类型图像的专家使用模式

```python
# 准备不同类型的图像
prompts_list = {
    "natural_scene": [
        {"prompt": "Describe this.", "multi_modal_data": {"image": "forest.jpg"}},
        {"prompt": "What do you see?", "multi_modal_data": {"image": "beach.jpg"}},
    ],
    "text_in_image": [
        {"prompt": "Read the text.", "multi_modal_data": {"image": "sign.jpg"}},
        {"prompt": "What does it say?", "multi_modal_data": {"image": "book.jpg"}},
    ],
    "diagram": [
        {"prompt": "Explain this diagram.", "multi_modal_data": {"image": "chart.jpg"}},
        {"prompt": "Analyze this.", "multi_modal_data": {"image": "graph.jpg"}},
    ]
}

# 分析每种类型
results = {}
for category, prompts in prompts_list.items():
    logits, _ = hs.get_moe_router_logits_generate(
        llm, prompts=prompts, max_tokens=30
    )
    analysis = hs.analyze_expert_usage(logits, top_k=2, normalize=True)
    results[category] = analysis

# 对比专家使用
print("Expert specialization by image type (Layer 10):")
print(f"{'Expert':<10} {'Natural':<12} {'Text':<12} {'Diagram':<12}")
for expert_id in range(8):
    natural = results['natural_scene']['expert_usage_counts'][10][expert_id]
    text = results['text_in_image']['expert_usage_counts'][10][expert_id]
    diagram = results['diagram']['expert_usage_counts'][10][expert_id]
    print(f"{expert_id:<10} {natural:<12.3f} {text:<12.3f} {diagram:<12.3f}")
```

## 高级功能

### 传递额外的生成参数

```python
# 支持所有vLLM的generate参数
router_logits, outputs = hs.get_moe_router_logits_generate(
    llm,
    prompts=multimodal_prompts,
    max_tokens=50,
    temperature=0.7,     # 生成温度
    top_p=0.9,           # nucleus sampling
    top_k=50,            # top-k sampling (注意：这是生成的top_k，不是MoE的)
    presence_penalty=0.1,
    frequency_penalty=0.1,
)
```

### 结合样本划分分析

```python
# 批量多模态输入，按样本划分
prompts = [
    {"prompt": "Describe this.", "multi_modal_data": {"image": f"image{i}.jpg"}}
    for i in range(5)
]

router_logits, outputs = hs.get_moe_router_logits_generate(
    llm,
    prompts=prompts,
    max_tokens=20,
    split_by_samples=True
)

# 逐个样本分析
for sample_idx, sample_logits in enumerate(router_logits):
    print(f"\n样本 {sample_idx}:")
    analysis = hs.analyze_expert_usage(
        sample_logits,
        top_k=2,
        normalize=True
    )
    print(f"  Token数: {analysis['num_tokens_analyzed']}")
    print(f"  Layer 10专家使用: {analysis['expert_usage_counts'][10]}")
```

## 注意事项

### 1. 模型兼容性
- 仅适用于支持多模态的vLLM模型
- 常见支持的模型：Qwen-VL系列、LLaVA系列等
- 模型必须使用`task="generate"`（默认）而不是`task="embed"`

### 2. 输入格式
- 多模态输入格式依赖于vLLM的API
- 不同版本的vLLM可能有细微差异
- 参考vLLM官方文档确认最新格式

### 3. 图像处理
- 图像可以是：
  - PIL.Image对象
  - 本地文件路径
  - URL（如果模型支持）
  - Base64编码（取决于vLLM版本）

### 4. 性能考虑
- 多模态输入通常比纯文本慢
- 图像编码会产生额外的tokens
- 建议使用较小的batch size

## 故障排除

### 问题1: 不支持多模态格式
```
ValueError: Unsupported input format
```
**解决**: 检查vLLM版本和模型是否支持多模态，参考vLLM文档

### 问题2: 图像加载失败
```
FileNotFoundError: Image not found
```
**解决**: 确认图像路径正确，或使用PIL预加载

### 问题3: Token数量不匹配
```
RuntimeWarning: Token lengths don't match
```
**解决**: 多模态输入的token数量计算可能有偏差，这是正常的，系统会自动调整

## 参考资源

- [vLLM多模态文档](https://docs.vllm.ai/)
- [Qwen-VL模型文档](https://github.com/QwenLM/Qwen-VL)
- [MoE Router Logits基础用法](./MOE_CAPTURE_USAGE.md)

