# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the efficiency benchmarks.

Every benchmark generates greedy completions for the same MATH prompts
on DeepSeek-R1-Distill-Qwen-1.5B and reports seconds per request and
output tokens per second.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "/home/shenyl/hf/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/"  # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
SEAL_VECTOR = os.path.join(
    HERE, "..", "..", "replications", "seal", "execution_avg_vector.gguf"
)
N_SEQUENTIAL = 10


def load_examples(n):
    """First `n` MATH training prompts in the R1 reasoning format."""
    with open(os.path.join(HERE, "..", "math", "math_train_1000.json"),
              encoding="utf-8") as f:
        problems = json.load(f)
    if n > len(problems):
        raise ValueError(f"requested {n} problems, only {len(problems)} available")
    return [
        "Please reason step by step, and put your final answer within "
        "\\boxed{}.\nUser: " + p + "\nAssistant: <think>"
        for p in problems[:n]
    ]


def report(total_output_tokens, elapsed, n_requests, ftl_s=None):
    """Paper metrics: FTL (ms), TPS (tok/s), TTLT (s)."""
    if ftl_s is not None:
        print(f"FTL:  {ftl_s * 1000:.2f} ms")
    print(f"TPS:  {total_output_tokens / elapsed:.2f} tok/s")
    print(f"TTLT: {elapsed / n_requests:.4f} s")
