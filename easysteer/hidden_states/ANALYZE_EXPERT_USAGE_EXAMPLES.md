# analyze_expert_usage 使用示例

`analyze_expert_usage`函数用于分析MoE模型的专家使用模式，支持区间分析和频率归一化。

## 基本用法

### 1. 分析所有tokens的专家使用

```python
import easysteer.hidden_states as hs
from vllm import LLM

llm = LLM(model="mistralai/Mixtral-8x7B-v0.1", task="embed")
router_logits, outputs = hs.get_moe_router_logits(llm, ["Hello world!"])

# 分析所有tokens
analysis = hs.analyze_expert_usage(router_logits, top_k=2)

# 查看第10层的专家使用次数
print(f"Layer 10 expert usage: {analysis['expert_usage_counts'][10]}")
# 输出: [120, 85, 95, 110, 78, 92, 105, 115]  # 8个专家的使用次数

print(f"Analyzed {analysis['num_tokens_analyzed']} tokens")
```

### 2. 获取激活频率（归一化）

```python
# 使用normalize=True获取激活频率（0.0-1.0）
analysis_freq = hs.analyze_expert_usage(router_logits, top_k=2, normalize=True)

# 频率表示每个专家在多少比例的tokens中被激活
print(f"Expert activation frequencies in layer 10:")
total_freq = 0
for expert_id, freq in enumerate(analysis_freq['expert_usage_counts'][10]):
    print(f"  Expert {expert_id}: {freq:.3f}")
    total_freq += freq

print(f"Total frequency sum: {total_freq:.3f}")  # 应该 ≈ top_k (2.0)

# 输出示例 (top_k=2):
#   Expert 0: 0.65  # 在65%的tokens中被激活
#   Expert 1: 0.42  # 在42%的tokens中被激活
#   Expert 2: 0.38
#   ...
#   Expert 7: 0.25
# Total frequency sum: 2.00  # 因为每个token选2个专家
```

## 区间分析

### 3. 只分析prompt部分

```python
# 假设prompt有15个tokens
prompt_length = 15

# 只分析prompt部分 (tokens 0-15)
analysis_prompt = hs.analyze_expert_usage(
    router_logits,
    top_k=2,
    token_range=(0, prompt_length),
    normalize=True
)

print(f"Prompt部分专家使用频率:")
for expert_id, freq in enumerate(analysis_prompt['expert_usage_counts'][10]):
    print(f"  Expert {expert_id}: {freq:.3f}")
```

### 4. 只分析生成部分

```python
# 假设生成从第15个token开始
generation_start = 15

# 只分析生成部分 (tokens 15到结束)
analysis_gen = hs.analyze_expert_usage(
    router_logits,
    top_k=2,
    token_range=(generation_start, None),  # None表示到结尾
    normalize=True
)

print(f"Generation部分专家使用频率:")
print(analysis_gen['expert_usage_counts'][10])
```

### 5. 分析特定区间

```python
# 分析中间某个区间 (tokens 10-20)
analysis_middle = hs.analyze_expert_usage(
    router_logits,
    top_k=2,
    token_range=(10, 20),
    normalize=False  # 绝对计数
)

print(f"Tokens 10-20的专家使用:")
print(f"Analyzed {analysis_middle['num_tokens_analyzed']} tokens")
print(analysis_middle['expert_usage_counts'][10])
```

## 高级应用

### 6. 对比prompt和generation的专家使用差异

```python
# 获取带样本划分的logits
router_logits, outputs = hs.get_moe_router_logits_generate(
    llm,
    prompts=["What is AI?"],
    max_tokens=10,
    split_by_samples=False
)

# 假设prompt长度为5
prompt_len = 5

# 分析prompt部分
prompt_analysis = hs.analyze_expert_usage(
    router_logits, 
    top_k=2, 
    token_range=(0, prompt_len),
    normalize=True
)

# 分析generation部分
gen_analysis = hs.analyze_expert_usage(
    router_logits,
    top_k=2,
    token_range=(prompt_len, None),
    normalize=True
)

# 对比第10层的专家使用
print("Layer 10 expert usage comparison:")
print(f"{'Expert':<10} {'Prompt':<12} {'Generation':<12} {'Diff':<12}")
for i in range(8):  # Mixtral有8个专家
    prompt_freq = prompt_analysis['expert_usage_counts'][10][i]
    gen_freq = gen_analysis['expert_usage_counts'][10][i]
    diff = gen_freq - prompt_freq
    print(f"{i:<10} {prompt_freq:<12.3f} {gen_freq:<12.3f} {diff:+.3f}")
```

### 7. 可视化专家使用模式

```python
import matplotlib.pyplot as plt
import numpy as np

# 获取所有层的专家使用频率
analysis = hs.analyze_expert_usage(router_logits, top_k=2, normalize=True)

# 提取数据
num_layers = len(analysis['expert_usage_counts'])
num_experts = len(analysis['expert_usage_counts'][0])

# 创建矩阵: [layer, expert]
usage_matrix = np.array([
    analysis['expert_usage_counts'][layer_id] 
    for layer_id in sorted(analysis['expert_usage_counts'].keys())
])

# 绘制热图
plt.figure(figsize=(12, 8))
plt.imshow(usage_matrix, aspect='auto', cmap='viridis')
plt.colorbar(label='Activation Frequency')
plt.xlabel('Expert ID')
plt.ylabel('Layer ID')
plt.title('Expert Activation Frequency Across Layers')
plt.tight_layout()
plt.savefig('expert_usage_heatmap.png')
```

### 8. 分析负载均衡

```python
# 获取负载均衡指标
analysis = hs.analyze_expert_usage(router_logits, top_k=2, normalize=True)

# 负载均衡指标 (coefficient of variation: lower is better)
print("Load balance metrics (CV) per layer:")
for layer_id in sorted(analysis['load_balance'].keys()):
    cv = analysis['load_balance'][layer_id]
    print(f"  Layer {layer_id:2d}: {cv:.3f}")

# CV接近0表示负载均衡良好
# CV较大表示某些专家负载过高
```

### 9. 识别专家专业化

```python
# 对不同类型的输入分析专家使用
math_prompts = ["What is 2+2?", "Solve x^2=4"]
code_prompts = ["def hello():", "class MyClass:"]

# 数学类prompt
math_logits, _ = hs.get_moe_router_logits(llm, math_prompts)
math_analysis = hs.analyze_expert_usage(math_logits, top_k=2, normalize=True)

# 代码类prompt
code_logits, _ = hs.get_moe_router_logits(llm, code_prompts)
code_analysis = hs.analyze_expert_usage(code_logits, top_k=2, normalize=True)

# 对比第10层
print("Expert specialization in layer 10:")
print(f"{'Expert':<10} {'Math':<12} {'Code':<12} {'Specialization':<15}")
for i in range(8):
    math_freq = math_analysis['expert_usage_counts'][10][i]
    code_freq = code_analysis['expert_usage_counts'][10][i]
    
    # 计算专业化程度
    if math_freq + code_freq > 0:
        specialization = abs(math_freq - code_freq) / (math_freq + code_freq)
    else:
        specialization = 0
    
    dominant = "Math" if math_freq > code_freq else "Code"
    print(f"{i:<10} {math_freq:<12.3f} {code_freq:<12.3f} {specialization:.3f} ({dominant})")
```

## 参数说明

### `token_range`
- `None` (默认): 分析所有tokens
- `(start, end)`: 分析tokens[start:end]
- `(start, None)`: 分析tokens[start:]到结尾
- `(None, end)` 或 `(0, end)`: 分析tokens[:end]

### `normalize`
- `False` (默认): 返回绝对计数
- `True`: 返回激活频率
  - 公式: `frequency = count / num_tokens`
  - 范围: [0.0, top_k]
  - 表示每个专家在多少比例的tokens中被激活
  - 所有专家的频率总和 ≈ top_k
  - 例如: 如果frequency=0.5且有100个tokens，表示该专家在50个tokens中被选中

### `top_k`
- 根据模型选择:
  - Mixtral: `top_k=2`
  - DeepSeek-V2: `top_k=6`
  - 其他: 查看模型配置

## 归一化公式详解

当使用`normalize=True`时:

```
frequency = count / num_tokens
```

**参数说明**:
- `count`: 该专家被选中的次数（绝对计数）
- `num_tokens`: 分析的token总数
- `frequency`: 该专家在多少比例的tokens中被激活

**示例计算**:
- 假设top_k=2，分析了100个tokens
- Expert 0被选中60次 → frequency = 60/100 = 0.60（在60%的tokens中被激活）
- Expert 1被选中45次 → frequency = 45/100 = 0.45（在45%的tokens中被激活）
- ...
- 所有专家频率总和 = (60+45+...)/100 = (100×2)/100 = 2.0 ≈ top_k

**关键性质**:
- 频率范围: [0.0, 1.0] 对于单个专家
- 所有专家频率总和 ≈ top_k（因为每个token选top_k个专家）
- frequency × num_tokens = count（可以还原绝对计数）

**与概率的区别**:
- 这**不是**专家被选中的概率（那个会总和为1.0）
- 这是激活频率，表示在多少token中该专家被激活
- 更直观地反映专家负载

## 返回值说明

```python
{
    'expert_selection': Dict[layer_id, Tensor],      # 选中的专家ID
    'expert_weights': Dict[layer_id, Tensor],        # 专家权重
    'expert_usage_counts': Dict[layer_id, List],     # 使用次数/频率
    'load_balance': Dict[layer_id, float],           # 负载均衡指标
    'num_tokens_analyzed': int,                      # 分析的token数量
}
```

## 注意事项

1. **区间选择**: 确保`token_range`不超出实际token数量
2. **归一化理解**: 归一化后的值是激活频率，不是概率权重
3. **多样本分析**: 如果使用`split_by_samples=False`，logits是concat的，需要手动计算各样本的token范围
4. **负载均衡**: CV (coefficient of variation) < 0.5 通常表示负载均衡良好

