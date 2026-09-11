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

Fixed output lengths keep token counts comparable across execution modes.
Routing correctness is covered by the e2e suites.
"""

import argparse
import os
import subprocess
import sys
import time

from common import (
    build_engine,
    distinct_spec,
    load_examples,
    nonnegative_int,
    positive_int,
    warmup,
)

MODES = ("eager", "split", "in_graph")


def run_mode(args):
    from vllm import SamplingParams

    llm = build_engine(
        args.mode, args.max_steer, max_num_seqs=args.max_num_seqs
    )
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
    prompts = load_examples(args.batch)
    layers = list(range(args.layers))
    ks = sorted(set(args.configs))
    specs = [distinct_spec(i, layers) for i in range(max(ks) or 1)]

    for k in ks:
        steering = None if k == 0 else [specs[i % k] for i in range(args.batch)]
        warmup(llm, prompts, steering)
        start = time.perf_counter()
        outs = llm.generate(prompts, params, steering=steering, use_tqdm=False)
        elapsed = time.perf_counter() - start
        total = sum(len(o.outputs[0].token_ids) for o in outs)
        print(
            f"RESULT mode={args.mode:8s} K={k:5d} "
            f"TPS={total / elapsed:9.2f} tok/s "
            f"TTLT={elapsed / args.batch:.4f} s "
            f"elapsed={elapsed:.4f} s tokens={total}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=positive_int, default=64)
    parser.add_argument(
        "--configs",
        type=nonnegative_int,
        nargs="+",
        default=[0, 1, 8, 32],
        help="K values: distinct configs per batch (0 = unsteered baseline)",
    )
    parser.add_argument(
        "--max-steer",
        type=positive_int,
        default=32,
        help="max_steer_vectors, identical for every tier (K <= this)",
    )
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument(
        "--max-num-seqs",
        type=positive_int,
        default=None,
        help="scheduler sequence limit; unset uses the vLLM default",
    )
    parser.add_argument(
        "--layers", type=positive_int, default=28, help="steered layer count per config"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["eager", "in_graph"],
        choices=MODES,
        help="tiers to compare (add 'split' for the piecewise middle tier)",
    )
    parser.add_argument(
        "--mode", choices=MODES, help=argparse.SUPPRESS
    )  # internal: child runs one tier
    args = parser.parse_args()

    ks = sorted(set(args.configs))
    if max(ks) > min(args.max_steer, args.batch):
        parser.error("K must not exceed --max-steer or --batch")

    if args.mode:
        run_mode(args)
        return

    passthrough = [
        "--batch",
        str(args.batch),
        "--configs",
        *[str(k) for k in ks],
        "--max-steer",
        str(args.max_steer),
        "--max-tokens",
        str(args.max_tokens),
        "--layers",
        str(args.layers),
        *(
            ["--max-num-seqs", str(args.max_num_seqs)]
            if args.max_num_seqs is not None
            else []
        ),
    ]
    env = {**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"}
    for mode in args.modes:
        print(f"===== tier: {mode} =====", flush=True)
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--mode", mode, *passthrough],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
