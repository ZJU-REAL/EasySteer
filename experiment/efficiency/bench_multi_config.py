# SPDX-License-Identifier: Apache-2.0
"""Mixed steering configurations in one batch: accuracy cost and scaling.

A batch of N requests is split across K distinct steering
configurations (K=0 means the unsteered baseline; K=1 is every request
sharing one config; K=N is one config per request). Every config is a
zero-scale vector at a distinct scale-irrelevant identity — the
generated text stays identical to the baseline, so the measured deltas
are pure steering-machinery overhead: per-request routing, slot
assignment, and per-token row tables.

--max-steer sweeps max_steer_vectors to show how the slot capacity
scales: it sizes the in-graph family tables (and the slot pool in
split mode), so raising it trades memory for concurrent-config
capacity. K distinct configs need K slots in flight.

The zero-scale byte-equality check runs only on eager engines at
batch <= 16: vLLM is not batch-deterministic at larger batch shapes
(identical unsteered runs differ from themselves — see
docs/features/batch_invariance.md; verified on both compiled batch-32
and eager batch-1000 runs), so byte oracles are meaningless there.
Routing correctness at scale is covered by tests/e2e/test_routing.py
(trace isolation) and tests/e2e/test_trigger_positions.py (co-batched
twins).
"""

import argparse
import os
import shutil
import tempfile
import time

from common import MODEL, SEAL_VECTOR, load_examples, report

from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec


def distinct_spec(i, layers, source=SEAL_VECTOR):
    """Config #i: zero-scale, so outputs match the baseline exactly,
    but a distinct fingerprint (a distinct exclude position — or a
    distinct source path in --distinct-paths mode)."""
    return SteeringSpec(vectors=[
        VectorSpec(
            source=source,
            scale=0.0,
            layers=layers,
            apply=ApplySpec(prompt="all", generation="all",
                            exclude_prompt_positions=[i]),
        )
    ])


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
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--configs", type=int, nargs="+",
                        default=[0, 1, 2, 4, 8],
                        help="K values: distinct configs per batch "
                             "(0 = unsteered baseline)")
    parser.add_argument("--max-steer", type=int, default=8,
                        help="max_steer_vectors (slot capacity; K <= this)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--distinct-paths", action="store_true",
                        help="give every config its own vector file on "
                             "disk (cold-load cost included; a second "
                             "pass measures warm reuse)")
    parser.add_argument("--layers", type=int, default=28,
                        help="steered layer count per config")
    parser.add_argument("--cudagraph", action="store_true",
                        help="compiled engine (in-graph steering tier)")
    args = parser.parse_args()

    layers = list(range(args.layers))
    ks = sorted(set(args.configs))
    assert all(k <= args.max_steer for k in ks), (
        "K distinct configs need K steering slots: raise --max-steer"
    )

    llm = LLM(
        model=MODEL,
        enable_steer_vector=True,
        steer_algorithms=["direct"],
        max_steer_vectors=args.max_steer,
        enforce_eager=not args.cudagraph,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens,
                            ignore_eos=True)
    prompts = load_examples(args.batch)

    tmpdir = None
    if args.distinct_paths:
        tmpdir = tempfile.mkdtemp(prefix="bench_steer_vecs_")
        paths = materialize_paths(max(ks) or 1, tmpdir)
        specs = [distinct_spec(i, layers, source=paths[i])
                 for i in range(max(ks) or 1)]
    else:
        specs = [distinct_spec(i, layers) for i in range(max(ks) or 1)]

    try:
        baseline_text = None
        for k in ks:
            if k == 0:
                steering = None
            else:
                # Round-robin the K configs across the batch.
                steering = [specs[i % k] for i in range(args.batch)]
            passes = (("cold", "warm") if args.distinct_paths and k
                      else ("",))
            for tag in passes:
                start = time.perf_counter()
                outs = llm.generate(prompts, params, steering=steering,
                                    use_tqdm=False)
                elapsed = time.perf_counter() - start
                texts = [o.outputs[0].text for o in outs]
                if args.cudagraph or args.batch > 16:
                    pass  # not batch-deterministic; see module docstring
                elif baseline_text is None:
                    baseline_text = texts
                else:
                    assert texts == baseline_text, (
                        f"zero-scale configs changed outputs at K={k} — "
                        "routing applied the wrong config to some request"
                    )
                total = sum(len(o.outputs[0].token_ids) for o in outs)
                label = f"K={k:5d}" + (f" {tag:4s}" if tag else "     ")
                print(f"{label} | ", end="")
                report(total, elapsed, args.batch)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
