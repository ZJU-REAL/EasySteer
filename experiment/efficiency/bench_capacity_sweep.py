# SPDX-License-Identifier: Apache-2.0
"""How max_steer_vectors shapes throughput on an all-distinct workload.

N requests, every one carrying a *different* zero-scale configuration,
submitted in one generate call. max_steer_vectors is a scheduling
constraint (like max_loras): at most `capacity` differently-configured
requests run concurrently, the rest wait for a slot. Sweeping the
capacity over the same workload exposes the trade-off directly:

  small capacity  -> little parallelism (throughput bounded by a
                     capacity-sized running batch), but each step pays
                     for few distinct configs;
  large capacity  -> full parallelism, but every step carries many
                     distinct configs (per-slot host cost) and, on the
                     in-graph tier, larger persistent tables.

Capacity is fixed at engine construction, so each sweep point boots
its own engine in a subprocess. Wall time includes the queueing the
constraint introduces — the metric is end-to-end drain throughput of
the whole workload.
"""

import argparse
import os
import subprocess
import sys
import time

from common import build_engine, distinct_spec, load_examples, positive_int, warmup


def run_capacity(args):
    from vllm import SamplingParams

    llm = build_engine(args.tier, args.capacity)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
    prompts = load_examples(args.batch)
    layers = list(range(args.layers))
    steering = [distinct_spec(i, layers) for i in range(args.batch)]

    warmup(llm, prompts, steering)
    start = time.perf_counter()
    outs = llm.generate(prompts, params, steering=steering, use_tqdm=False)
    elapsed = time.perf_counter() - start
    total = sum(len(o.outputs[0].token_ids) for o in outs)
    print(
        f"RESULT tier={args.tier:8s} capacity={args.capacity:5d} "
        f"TPS={total / elapsed:9.2f} tok/s TTLT={elapsed / args.batch:.4f} s "
        f"elapsed={elapsed:.4f} s tokens={total}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        type=positive_int,
        default=256,
        help="requests, each with a distinct config",
    )
    parser.add_argument(
        "--capacities", type=positive_int, nargs="+", default=[2, 8, 32, 128, 256]
    )
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument("--layers", type=positive_int, default=28)
    parser.add_argument("--tier", choices=["in_graph", "eager"], default="in_graph")
    parser.add_argument(
        "--capacity", type=positive_int, help=argparse.SUPPRESS
    )  # internal: child runs one
    args = parser.parse_args()

    capacities = [args.capacity] if args.capacity is not None else args.capacities
    if max(capacities) > args.batch:
        parser.error("capacity must not exceed --batch")
    if args.capacity is not None:
        run_capacity(args)
        return

    env = {**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"}
    for capacity in dict.fromkeys(args.capacities):
        subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--capacity",
                str(capacity),
                "--batch",
                str(args.batch),
                "--max-tokens",
                str(args.max_tokens),
                "--layers",
                str(args.layers),
                "--tier",
                args.tier,
            ],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
