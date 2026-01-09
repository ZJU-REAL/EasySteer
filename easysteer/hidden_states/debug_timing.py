# 临时诊断脚本：定位MoE logits capture的瓶颈

import time
import easysteer.hidden_states as hs
from vllm import LLM

# 创建模型
print("Loading model...")
llm = LLM(
    model="Qwen/Qwen3-VL-30B-A3B-Thinking",  # 替换成你的模型
    tensor_parallel_size=4,
    trust_remote_code=True,
)

# 测试4000 tokens
print("\n=== Testing 4000 tokens ===")
prompt_4k = "A" * 4000  # 简单文本，约4000 tokens

t0 = time.time()
print(f"[{time.time()-t0:.2f}s] Starting capture...")

router_logits, outputs = hs.get_moe_router_logits_generate(
    llm,
    prompts=[prompt_4k],
    max_tokens=1
)

t1 = time.time()
print(f"[{t1-t0:.2f}s] Capture completed")
print(f"Total time: {t1-t0:.2f}s")
print(f"Captured {len(router_logits)} layers")

# 测试8000 tokens
print("\n=== Testing 8000 tokens ===")
prompt_8k = "B" * 8000  # 约8000 tokens

t0 = time.time()
print(f"[{time.time()-t0:.2f}s] Starting capture...")

router_logits, outputs = hs.get_moe_router_logits_generate(
    llm,
    prompts=[prompt_8k],
    max_tokens=1
)

t1 = time.time()
print(f"[{t1-t0:.2f}s] Capture completed")
print(f"Total time: {t1-t0:.2f}s")
print(f"Captured {len(router_logits)} layers")

