#!/usr/bin/env python3
"""Verify steering correctness for your model/hardware before running experiments.

Runs four comparisons and reports pass/fail for each:

  1. plain vLLM (chunked prefill ON)  vs  plain vLLM (chunked prefill OFF)
     → Confirms disabling chunked prefill is safe for your model.

  2. plain vLLM (chunked prefill OFF) vs  steered at scale=0 (chunked prefill OFF)
     → Confirms the steering infrastructure is transparent when inactive.

  3. steered eager (scale=4)  vs  steered CUDA graphs (scale=4)
     → Confirms CUDA graphs don't skip or corrupt the steering intervention.

  4. steered scale=4  vs  steered scale=0
     → Confirms steering actually changes the output (sanity check).

Run this script on your target model and hardware BEFORE running any
experiments that depend on steering correctness.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/verify_steering_correctness.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --vector vectors/happy_diffmean.gguf \
        --target-layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
        --scale 4.0

All arguments have sensible defaults for the Qwen2.5-1.5B + happy vector setup.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

LOGPROB_ATOL = 1e-4

PROMPTS_RAW = [
    "Describe a rainy Monday morning.",
    "What happens when you find a lost cat?",
    "Tell me about riding a bicycle through the park.",
    "What's the best way to make coffee?",
    "Describe the view from a mountaintop at sunrise.",
]


def build_prompts(model: str) -> list[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS_RAW
    ]


def extract_logprobs(
    outputs,
) -> list[tuple[list[str], list[float]]]:
    results = []
    for out in outputs:
        comp = out.outputs[0]
        tokens: list[str] = []
        lps: list[float] = []
        for i, lp_dict in enumerate(comp.logprobs):
            tid = comp.token_ids[i]
            entry = lp_dict[tid]
            tokens.append(entry.decoded_token or f"<id:{tid}>")
            lps.append(entry.logprob)
        results.append((tokens, lps))
    return results


def generate(
    llm: LLM,
    prompts: list[str],
    max_tokens: int,
    steer_req: SteerVectorRequest | None = None,
) -> list[tuple[list[str], list[float]]]:
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=0)
    kwargs = {}
    if steer_req is not None:
        kwargs["steer_vector_request"] = steer_req
    outputs = llm.generate(prompts, sampling_params=sampling, **kwargs)
    return extract_logprobs(outputs)


def compare(
    name_a: str,
    results_a: list[tuple[list[str], list[float]]],
    name_b: str,
    results_b: list[tuple[list[str], list[float]]],
    expect_same: bool,
) -> tuple[bool, float]:
    """Compare two sets of results.

    If expect_same=True, checks tokens match and logprobs are within LOGPROB_ATOL.
    If expect_same=False, checks that at least one prompt differs (sanity check).
    Returns (passed, global_max_diff).
    """
    global_max = 0.0
    all_match = True
    for i in range(len(results_a)):
        toks_a, lps_a = results_a[i]
        toks_b, lps_b = results_b[i]
        if toks_a != toks_b:
            all_match = False
        max_diff = max(
            (abs(a - b) for a, b in zip(lps_a, lps_b)),
            default=0.0,
        )
        global_max = max(global_max, max_diff)

    if expect_same:  # noqa: SIM108
        passed = all_match and global_max <= LOGPROB_ATOL
    else:
        passed = not all_match  # at least one prompt should differ
    return passed, global_max


def make_llm(
    model: str,
    gpu_mem: float,
    max_model_len: int,
    *,
    enable_steer: bool = False,
    enforce_eager: bool = True,
    chunked_prefill: bool = False,
    vector_path: str | None = None,
    scale: float = 0.0,
    target_layers: list[int] | None = None,
    normalize: bool = True,
) -> LLM:
    kwargs: dict = dict(
        model=model,
        enforce_eager=enforce_eager,
        enable_chunked_prefill=chunked_prefill,
        enable_prefix_caching=False,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
    )
    if vector_path is not None:
        # Server-level steering
        kwargs["steer_vector_path"] = vector_path
        kwargs["steer_scale"] = scale
        kwargs["steer_target_layers"] = target_layers
        kwargs["steer_normalize"] = normalize
        kwargs["steer_algorithm"] = "direct"
        # Server-level implies enable_steer_vector
    elif enable_steer:
        kwargs["enable_steer_vector"] = True
    return LLM(**kwargs)


def run_check(
    name: str,
    name_a: str,
    results_a: list[tuple[list[str], list[float]]],
    name_b: str,
    results_b: list[tuple[list[str], list[float]]],
    expect_same: bool,
) -> bool:
    passed, max_diff = compare(name_a, results_a, name_b, results_b, expect_same)
    if expect_same:
        status = "PASS" if passed else "FAIL"
        detail = f"max_diff={max_diff:.6f} (tol={LOGPROB_ATOL})"
    else:
        status = "PASS" if passed else "FAIL"
        detail = (
            "outputs differ"
            if passed
            else "outputs are IDENTICAL (steering had no effect!)"
        )
    print(f"  [{status}] {name}: {detail}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify steering correctness for your model/hardware.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--vector", required=True,
        help="Path to steering vector (.gguf)",
    )
    parser.add_argument(
        "--target-layers", type=int, nargs="+", required=True,
        help="Layer indices to steer",
    )
    parser.add_argument(
        "--scale", type=float, default=4.0,
        help="Steering scale for comparison",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=40,
        help="Tokens to generate per prompt",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    args = parser.parse_args()

    vector_path = str(Path(args.vector).resolve())
    if not Path(vector_path).exists():
        print(f"ERROR: Vector file not found: {vector_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Steering Correctness Verification")
    print("=" * 60)
    print(f"  Model:         {args.model}")
    print(f"  Vector:        {vector_path}")
    print(f"  Target layers: {args.target_layers}")
    print(f"  Scale:         {args.scale}")
    print(f"  Max tokens:    {args.max_tokens}")
    print(f"  Tolerance:     {LOGPROB_ATOL}")
    print()

    prompts = build_prompts(args.model)
    common = dict(
        model=args.model,
        gpu_mem=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # ── 1. Plain chunked ON ──────────────────────────────────────
    print("[1/5] Plain vLLM, chunked_prefill=True")
    llm = make_llm(**common, chunked_prefill=True)
    plain_chunked = generate(llm, prompts, args.max_tokens)
    del llm
    torch.cuda.empty_cache()

    # ── 2. Plain chunked OFF ─────────────────────────────────────
    print("[2/5] Plain vLLM, chunked_prefill=False")
    llm = make_llm(**common, chunked_prefill=False)
    plain_nochunk = generate(llm, prompts, args.max_tokens)
    del llm
    torch.cuda.empty_cache()

    # ── 3. Steered scale=0, eager ────────────────────────────────
    print("[3/5] Steered vLLM, scale=0, eager")
    llm = make_llm(
        **common, vector_path=vector_path, scale=0.0,
        target_layers=args.target_layers, normalize=args.normalize,
    )
    steered_s0 = generate(llm, prompts, args.max_tokens)
    del llm
    torch.cuda.empty_cache()

    # ── 4. Steered scale=N, eager ────────────────────────────────
    print(f"[4/5] Steered vLLM, scale={args.scale}, eager")
    llm = make_llm(
        **common, vector_path=vector_path, scale=args.scale,
        target_layers=args.target_layers, normalize=args.normalize,
    )
    steered_eager = generate(llm, prompts, args.max_tokens)
    del llm
    torch.cuda.empty_cache()

    # ── 5. Steered scale=N, CUDA graphs ──────────────────────────
    print(f"[5/5] Steered vLLM, scale={args.scale}, CUDA graphs")
    llm = make_llm(
        **common, enforce_eager=False,
        vector_path=vector_path, scale=args.scale,
        target_layers=args.target_layers, normalize=args.normalize,
    )
    steered_cg = generate(llm, prompts, args.max_tokens)
    del llm
    torch.cuda.empty_cache()

    # ── Results ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    checks = []
    checks.append(run_check(
        "Chunked prefill ON vs OFF (plain, no steering)",
        "plain_chunked", plain_chunked,
        "plain_nochunk", plain_nochunk,
        expect_same=True,
    ))
    checks.append(run_check(
        "Plain vs scale=0 steering (both chunked OFF)",
        "plain_nochunk", plain_nochunk,
        "steered_s0", steered_s0,
        expect_same=True,
    ))
    checks.append(run_check(
        f"Steered eager vs CUDA graphs (scale={args.scale})",
        "steered_eager", steered_eager,
        "steered_cg", steered_cg,
        expect_same=True,
    ))
    checks.append(run_check(
        f"Steering has effect (scale={args.scale} vs scale=0)",
        "steered_eager", steered_eager,
        "steered_s0", steered_s0,
        expect_same=False,
    ))

    print()
    n_pass = sum(checks)
    n_total = len(checks)
    if all(checks):
        print(f"*** ALL {n_total} CHECKS PASSED — safe to run experiments ***")
    else:
        print(f"*** {n_total - n_pass}/{n_total} CHECKS FAILED ***")
        print("Do NOT run experiments until all checks pass.")
        sys.exit(1)


if __name__ == "__main__":
    main()
