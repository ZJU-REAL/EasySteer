# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the efficiency benchmarks.

Every benchmark generates greedy completions for the same MATH prompts
on DeepSeek-R1-Distill-Qwen-1.5B and reports seconds per request and
output tokens per second.
"""

import json
import os
from argparse import ArgumentTypeError
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = os.environ.get("EASYSTEER_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
SEAL_VECTOR = os.environ.get(
    "EASYSTEER_VECTOR",
    str(HERE.parent.parent / "replications/seal/execution_avg_vector.gguf"),
)
N_SEQUENTIAL = 10


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise ArgumentTypeError("must be nonnegative")
    return number


def load_examples(n: int) -> list[str]:
    """First `n` MATH training prompts in the R1 reasoning format."""
    with open(
        os.environ.get(
            "EASYSTEER_BENCH_DATA",
            str(HERE.parent / "math/math_train_1000.json"),
        ),
        encoding="utf-8",
    ) as f:
        problems = json.load(f)
    if not isinstance(problems, list) or not all(isinstance(p, str) for p in problems):
        raise ValueError("benchmark data must be a JSON list of problem strings")
    if n > len(problems):
        raise ValueError(f"requested {n} problems, only {len(problems)} available")
    return [
        "Please reason step by step, and put your final answer within "
        "\\boxed{}.\nUser: " + p + "\nAssistant: <think>"
        for p in problems[:n]
    ]


def build_engine(
    tier: str,
    max_steer: int | None = None,
    *,
    multi_vector: bool = False,
    max_num_seqs: int | None = None,
):
    """Use the same model and batching policy for all vLLM comparisons."""
    from vllm import LLM

    engine_kwargs = {
        "model": MODEL,
        "dtype": "bfloat16",
        "enable_steer_vector": True,
        "steer_algorithms": ["direct"],
        "steer_multi_vector": multi_vector,
        "max_steer_vectors": max_steer,
        "enforce_eager": tier == "eager",
        "steer_graph_mode": "auto" if tier == "eager" else tier,
    }
    if max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = max_num_seqs
    llm = LLM(**engine_kwargs)
    config = llm.llm_engine.vllm_config
    print(
        f"ENGINE dtype={config.model_config.dtype} "
        f"steer_graph_mode={config.steer_vector_config.graph_mode} "
        f"cudagraph_mode={config.compilation_config.cudagraph_mode} "
        f"prefix_caching={config.cache_config.enable_prefix_caching} "
        f"chunked_prefill={config.scheduler_config.enable_chunked_prefill} "
        f"max_num_batched_tokens={config.scheduler_config.max_num_batched_tokens}",
        flush=True,
    )
    return llm


def distinct_spec(index: int, layers: list[int], source: str = SEAL_VECTOR):
    """Distinct slot fingerprints with identical zero-scale token coverage.

    The generation window unions with ``generation="all"``, preserving the
    effective mask without referring to potentially short prompt positions.
    """
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    return SteeringSpec(
        vectors=[
            VectorSpec(
                source=source,
                scale=0.0,
                layers=layers,
                apply=ApplySpec(
                    prompt="all", generation="all", generation_window=(index, index + 1)
                ),
            )
        ]
    )


def warmup(llm, prompts, steering=None) -> None:
    """Exercise prefill, decode and the selected payloads outside the timer."""
    from vllm import SamplingParams

    llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
        steering=steering,
        use_tqdm=False,
    )
    reset_prefix_cache(llm)


def reset_prefix_cache(llm) -> None:
    """Clear KV prefix hashes after a completed warmup or probe call."""
    if not llm.reset_prefix_cache():
        raise RuntimeError("prefix-cache reset failed while requests were still running")


def report(total_output_tokens, elapsed, n_requests, one_token_s=None):
    """Report aggregate throughput and amortized time per submitted request."""
    if one_token_s is not None:
        print(f"One-token call: {one_token_s * 1000:.2f} ms")
    print(f"Elapsed: {elapsed:.4f} s; output tokens: {total_output_tokens}")
    print(f"TPS:  {total_output_tokens / elapsed:.2f} tok/s")
    print(f"TTLT: {elapsed / n_requests:.4f} s")
