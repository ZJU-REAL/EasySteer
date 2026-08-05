# SPDX-License-Identifier: Apache-2.0
"""Apples-to-apples steering-tier comparison under identical conditions.

Same model, same batch size, same prompts, same max_tokens and the
same K distinct zero-scale configurations round-robined over the batch
— the only variable is the steering execution tier:

  eager     enforce_eager engine, steering ops run as plain eager ops
  split     compiled engine, steering ops as splitting ops (piecewise
            cudagraphs)
  in_graph  compiled engine, steering baked into full cudagraphs as a
            data-driven kernel

Each tier runs in its own subprocess (engines cannot share the GPU) and
sweeps the same K values, so rows are directly comparable both across K
within a tier and across tiers at fixed K. K=0 is the tier's unsteered
baseline: the spread of the K=0 rows is the cost of the execution mode
itself; the decay over K within one row group is the cost of
distinct-configuration steering.

Outputs are not compared across tiers: compiled and eager execution are
not numerically identical (and vLLM is not batch-deterministic at these
batch shapes). Routing correctness is covered by the e2e suites.
"""

import argparse
import os
import subprocess
import sys
import time

from bench_multi_config import distinct_spec
from common import MODEL, load_examples

MODES = ("eager", "split", "in_graph")


def build_engine(mode, max_steer):
    from vllm import LLM

    kwargs = dict(
        model=MODEL,
        enable_steer_vector=True,
        steer_algorithms=["direct"],
        max_steer_vectors=max_steer,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    if mode == "eager":
        kwargs["enforce_eager"] = True
    else:
        kwargs["steer_graph_mode"] = mode
    return LLM(**kwargs)


def run_mode(args):
    from vllm import SamplingParams

    llm = build_engine(args.mode, args.max_steer)
    params = SamplingParams(
        temperature=0, max_tokens=args.max_tokens, ignore_eos=True
    )
    prompts = load_examples(args.batch)
    layers = list(range(args.layers))
    ks = sorted(set(args.configs))
    specs = [distinct_spec(i, layers) for i in range(max(ks) or 1)]

    # One untimed pass so compile/capture warmup never lands in a row.
    llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
        use_tqdm=False,
    )

    for k in ks:
        steering = (
            None if k == 0 else [specs[i % k] for i in range(args.batch)]
        )
        start = time.perf_counter()
        outs = llm.generate(prompts, params, steering=steering, use_tqdm=False)
        elapsed = time.perf_counter() - start
        total = sum(len(o.outputs[0].token_ids) for o in outs)
        print(
            f"RESULT mode={args.mode:8s} K={k:5d} "
            f"TPS={total / elapsed:9.2f} tok/s "
            f"TTLT={elapsed / args.batch:.4f} s",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--configs", type=int, nargs="+",
                        default=[0, 1, 8, 32],
                        help="K values: distinct configs per batch "
                             "(0 = unsteered baseline)")
    parser.add_argument("--max-steer", type=int, default=32,
                        help="max_steer_vectors, identical for every "
                             "tier (K <= this)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--layers", type=int, default=28,
                        help="steered layer count per config")
    parser.add_argument("--modes", nargs="+", default=["eager", "in_graph"],
                        choices=MODES,
                        help="tiers to compare (add 'split' for the "
                             "piecewise middle tier)")
    parser.add_argument("--mode", choices=MODES,
                        help=argparse.SUPPRESS)  # internal: child runs one tier
    args = parser.parse_args()

    ks = sorted(set(args.configs))
    assert all(k <= args.max_steer for k in ks), (
        "K distinct configs need K steering slots: raise --max-steer"
    )

    if args.mode:
        run_mode(args)
        return

    passthrough = [
        "--batch", str(args.batch),
        "--configs", *[str(k) for k in ks],
        "--max-steer", str(args.max_steer),
        "--max-tokens", str(args.max_tokens),
        "--layers", str(args.layers),
    ]
    env = {**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"}
    for mode in args.modes:
        print(f"===== tier: {mode} =====", flush=True)
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--mode", mode,
             *passthrough],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
