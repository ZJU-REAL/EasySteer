# SPDX-License-Identifier: Apache-2.0
"""Mixed steering configurations in one batch: intervention overhead and scaling.

A batch of N requests is split across K distinct steering
configurations (K=0 means the unsteered baseline; K=1 is every request
sharing one config; K=N is one config per request). Every config is a
zero-scale vector with a distinct fingerprint to measure steering
overhead: per-request routing, slot
assignment, and per-token row tables.

--max-steer fixes the slot capacity for the K sweep. Use
bench_capacity_sweep.py to vary capacity for an unchanged workload.

For eager batches <= 16, the script also reports how many outputs match
the unsteered control. This is an observation alongside throughput.
Routing correctness at scale is covered by tests/e2e/test_routing.py
(trace isolation) and tests/e2e/test_trigger_positions.py (co-batched
twins).
"""

import argparse
import os
import shutil
import tempfile
import time
from contextlib import nullcontext

from common import (
    SEAL_VECTOR,
    build_engine,
    distinct_spec,
    load_examples,
    nonnegative_int,
    positive_int,
    report,
    warmup,
)


def materialize_paths(n, tmpdir):
    """N distinct on-disk vector files (copies of the reference gguf):
    each distinct path is a distinct config that must be loaded from
    disk into its own slot — the cold-load and slot-management cost of
    serving many configurations."""
    paths = []
    for i in range(n):
        p = os.path.join(tmpdir, f"vec_{i:05d}.gguf")
        shutil.copyfile(SEAL_VECTOR, p)
        paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=positive_int, default=32)
    parser.add_argument(
        "--configs",
        type=nonnegative_int,
        nargs="+",
        default=[0, 1, 2, 4, 8],
        help="K values: distinct configs per batch (0 = unsteered baseline)",
    )
    parser.add_argument(
        "--max-steer",
        type=positive_int,
        default=8,
        help="max_steer_vectors (slot capacity; K <= this)",
    )
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument(
        "--max-num-seqs",
        type=positive_int,
        default=None,
        help="scheduler sequence limit; unset uses the vLLM default",
    )
    parser.add_argument(
        "--distinct-paths",
        action="store_true",
        help="give every config its own vector file on "
        "disk (cold-load cost included; a second "
        "pass measures warm reuse)",
    )
    parser.add_argument(
        "--layers", type=positive_int, default=28, help="steered layer count per config"
    )
    parser.add_argument(
        "--cudagraph",
        action="store_true",
        help="compiled engine (in-graph steering tier)",
    )
    args = parser.parse_args()

    layers = list(range(args.layers))
    ks = sorted(set(args.configs))
    if max(ks) > min(args.max_steer, args.batch):
        parser.error("K must not exceed --max-steer or --batch")

    from vllm import SamplingParams

    llm = build_engine(
        "in_graph" if args.cudagraph else "eager",
        args.max_steer,
        max_num_seqs=args.max_num_seqs,
    )
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
    prompts = load_examples(args.batch)

    temporary = (
        tempfile.TemporaryDirectory(prefix="bench_steer_vecs_")
        if args.distinct_paths
        else nullcontext(None)
    )
    with temporary as tmpdir:
        baseline_text = None
        for k in ks:
            if k == 0:
                steering = None
            else:
                if tmpdir is not None:
                    # Each K starts with genuinely new paths; earlier rows must
                    # not populate the payload cache for a later cold row.
                    directory = os.path.join(tmpdir, f"k_{k}")
                    os.mkdir(directory)
                    paths = materialize_paths(k, directory)
                    specs = [distinct_spec(i, layers, paths[i]) for i in range(k)]
                else:
                    specs = [distinct_spec(i, layers) for i in range(k)]
                # Round-robin the K configs across the batch.
                steering = [specs[i % k] for i in range(args.batch)]
            warmup(llm, prompts, None if args.distinct_paths else steering)
            passes = ("cold", "warm") if args.distinct_paths and k else ("",)
            for tag in passes:
                start = time.perf_counter()
                outs = llm.generate(prompts, params, steering=steering, use_tqdm=False)
                elapsed = time.perf_counter() - start
                texts = [o.outputs[0].text for o in outs]
                if k == 0:
                    baseline_text = texts
                elif (
                    not args.cudagraph
                    and args.batch <= 16
                    and baseline_text is not None
                ):
                    matches = sum(a == b for a, b in zip(texts, baseline_text))
                    print(
                        f"K={k}: {matches}/{len(texts)} texts match the unsteered control"
                    )
                total = sum(len(o.outputs[0].token_ids) for o in outs)
                label = f"K={k:5d}" + (f" {tag:4s}" if tag else "     ")
                print(f"{label} | ", end="")
                report(total, elapsed, args.batch)


if __name__ == "__main__":
    main()
