#!/usr/bin/env python3
"""Collect optional steering diagnostics, with one engine per child process.

Reports token equality and logprob differences for the same fixed-length batch
under five configurations. These observations are not pass/fail or a substitute
for the mechanism-level pytest suites: kernel and scheduler numerics can change
text even at temperature=0, including between eager and compiled runs.

Usage from the repository root:
    CUDA_VISIBLE_DEVICES=0 python tests/verify_steering_correctness.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --vector vectors/happy_diffmean.gguf --target-layers 10 11 12 \
        --output /tmp/steering-diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CASES = ("plain_chunked", "plain", "zero_eager", "steered_eager", "steered_graph")
PROMPTS = [
    "Alice's dog has passed away. Please comfort her.",
    "Describe a rainy Monday morning.",
    "What happens when you find a lost cat?",
    "Tell me about riding a bicycle through the park.",
    "Describe the view from a mountaintop at sunrise.",
]


def run_case(args: argparse.Namespace) -> dict:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in PROMPTS
    ]
    kwargs = dict(
        model=args.model,
        enforce_eager=args._case != "steered_graph",
        enable_chunked_prefill=args._case == "plain_chunked",
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=len(PROMPTS),
        max_num_batched_tokens=args.max_model_len,
    )
    if args._case.startswith(("zero_", "steered_")):
        kwargs["steering_config"] = SteeringSpec(
            vectors=[
                VectorSpec(
                    source=args.vector,
                    algorithm="direct",
                    scale=0.0 if args._case == "zero_eager" else args.scale,
                    layers=args.target_layers,
                    normalize=False,
                    apply=ApplySpec(prompt="all", generation="all"),
                )
            ]
        ).model_dump_json()
        if args._case == "steered_graph":
            kwargs["steer_graph_mode"] = "in_graph"
    llm = LLM(**kwargs)
    params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        logprobs=0,
        seed=0,
    )
    outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    results = []
    for output in outputs:
        completion = output.outputs[0]
        results.append(
            {
                "text": completion.text,
                "token_ids": list(completion.token_ids),
                "logprobs": [
                    distribution[token].logprob
                    for token, distribution in zip(
                        completion.token_ids, completion.logprobs
                    )
                ],
            }
        )
    config = llm.llm_engine.vllm_config
    return {
        "case": args._case,
        "gpu_name": torch.cuda.get_device_name(0),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
        "steer_graph_mode": getattr(config.steer_vector_config, "graph_mode", None),
        "outputs": results,
    }


def compare(left: dict, right: dict) -> dict:
    """Compare logprobs only along a shared autoregressive token prefix."""
    equal = 0
    shared_tokens = 0
    maximum = 0.0
    for a, b in zip(left["outputs"], right["outputs"], strict=True):
        equal += a["token_ids"] == b["token_ids"]
        for token_a, token_b, lp_a, lp_b in zip(
            a["token_ids"],
            b["token_ids"],
            a["logprobs"],
            b["logprobs"],
            strict=True,
        ):
            if token_a != token_b:
                break
            shared_tokens += 1
            maximum = max(maximum, abs(lp_a - lp_b))
    return {
        "left": left["case"],
        "right": right["case"],
        "identical_token_sequences": equal,
        "total_prompts": len(left["outputs"]),
        "shared_prefix_tokens": shared_tokens,
        "max_shared_prefix_logprob_difference": maximum if shared_tokens else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("STEER_TEST_MODEL"))
    parser.add_argument("--vector", required=True)
    parser.add_argument("--target-layers", type=int, nargs="+", required=True)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--output", type=Path, help="New JSON result path")
    parser.add_argument("--_case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.model:
        parser.error("provide --model or STEER_TEST_MODEL")
    args.vector = str(Path(args.vector).expanduser().resolve())
    if not Path(args.vector).is_file():
        parser.error(f"vector file does not exist: {args.vector}")
    if args.max_tokens <= 0 or args.max_tokens >= args.max_model_len:
        parser.error("--max-tokens must be positive and below --max-model-len")
    if args.output and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args._case:
        args._result.write_text(json.dumps(run_case(args), indent=2), encoding="utf-8")
        return

    observations = []
    with tempfile.TemporaryDirectory(prefix="easysteer-diagnostic-") as tmp:
        for case in CASES:
            result_path = Path(tmp) / f"{case}.json"
            print(f"Collecting {case} in a separate process", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--_case",
                    case,
                    "--_result",
                    str(result_path),
                ],
                check=True,
            )
            observations.append(json.loads(result_path.read_text(encoding="utf-8")))

    by_case = {observation["case"]: observation for observation in observations}
    comparisons = [
        compare(by_case[left], by_case[right])
        for left, right in (
            ("plain_chunked", "plain"),
            ("plain", "zero_eager"),
            ("steered_eager", "steered_graph"),
            ("zero_eager", "steered_eager"),
        )
    ]
    report = {
        "workload": vars(args) | {"output": str(args.output), "_result": None},
        "observations": observations,
        "comparisons": comparisons,
    }
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
    print(json.dumps(comparisons, indent=2))
    print(
        "Diagnostic collection completed. Use tests/run_suites.sh baseline for "
        "mechanism-level validation; text differences alone do not identify "
        "a steering regression."
    )


if __name__ == "__main__":
    main()
