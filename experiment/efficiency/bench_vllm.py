# SPDX-License-Identifier: Apache-2.0
"""EasySteer/vLLM efficiency benchmark (EasySteer paper, Section 5.1).

Steering configurations, all using zero scale to measure intervention overhead:
    baseline      - no steering
    single_layer  - one vector at one layer (20)
    all_layer     - one vector on all 28 layers
    multi_vector  - three sequential vectors on all 28 layers

Sequential mode times 10 single-prompt requests; --batch N submits N
prompts in one generate call (vLLM continuous batching). --max-tokens
matches the paper's two settings (128 and 2048). The one-token call is
timed separately after warmup; it is a batch drain time, not streaming TTFT.
"""

import argparse
import time

from common import (
    N_SEQUENTIAL,
    SEAL_VECTOR,
    build_engine,
    load_examples,
    nonnegative_int,
    report,
    warmup,
)


def zero_scale_spec(n_vectors, layers):
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    return SteeringSpec(
        conflict="sequential",
        vectors=[
            VectorSpec(
                source=SEAL_VECTOR,
                scale=0.0,
                layers=layers,
                apply=ApplySpec(prompt="all", generation="all"),
            )
            for _ in range(n_vectors)
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["baseline", "single_layer", "all_layer", "multi_vector"],
        default="baseline",
    )
    parser.add_argument(
        "--batch",
        type=nonnegative_int,
        default=0,
        help="batch size; 0 = sequential single requests",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, choices=[128, 2048])
    parser.add_argument(
        "--cudagraph",
        action="store_true",
        help="enable CUDA graphs (paper numbers are eager)",
    )
    parser.add_argument(
        "--graph-mode",
        choices=["split", "in_graph"],
        default=None,
        help="steering graph tier under --cudagraph: "
        "in_graph captures the steering kernel into the "
        "graph; split splits at steered layers "
        "(all algorithms)",
    )
    args = parser.parse_args()
    if args.graph_mode is not None and not args.cudagraph:
        parser.error("--graph-mode requires --cudagraph")
    if args.mode == "multi_vector" and args.graph_mode == "in_graph":
        parser.error("multi-vector steering requires --graph-mode split or auto")

    from vllm import SamplingParams

    steering = {
        "baseline": None,
        "single_layer": zero_scale_spec(1, [20]),
        "all_layer": zero_scale_spec(1, list(range(28))),
        "multi_vector": zero_scale_spec(3, list(range(28))),
    }[args.mode]
    tier = (args.graph_mode or "auto") if args.cudagraph else "eager"
    llm = build_engine(
        tier,
        multi_vector=args.mode == "multi_vector",
    )
    params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
        ignore_eos=True,
    )
    one_token = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)

    if args.batch:
        examples = load_examples(args.batch)
        warmup(llm, examples, steering)
        start = time.perf_counter()
        llm.generate(examples, one_token, steering=steering, use_tqdm=False)
        one_token_s = time.perf_counter() - start
        start = time.perf_counter()
        outs = llm.generate(examples, params, steering=steering, use_tqdm=False)
        elapsed = time.perf_counter() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        report(tokens, elapsed, args.batch, one_token_s=one_token_s)
    else:
        examples = load_examples(N_SEQUENTIAL)
        warmup(llm, examples[0], steering)
        start = time.perf_counter()
        llm.generate(examples[0], one_token, steering=steering, use_tqdm=False)
        one_token_s = time.perf_counter() - start
        tokens = 0
        start = time.perf_counter()
        for example in examples:
            outs = llm.generate(example, params, steering=steering, use_tqdm=False)
            tokens += len(outs[0].outputs[0].token_ids)
        elapsed = time.perf_counter() - start
        report(tokens, elapsed, len(examples), one_token_s=one_token_s)


if __name__ == "__main__":
    main()
