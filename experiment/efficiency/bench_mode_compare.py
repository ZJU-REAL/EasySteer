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

Each K value runs in its own subprocess (engines cannot share the GPU),
so rows are directly comparable without carrying graph or payload state
from one K value into the next. K=0 is the tier's unsteered baseline: the
spread of the K=0 rows is the cost of the execution mode itself; the decay
over K within one row group is the cost of distinct-configuration steering.

Fixed output lengths keep token counts comparable across execution modes.
Routing correctness is covered by the e2e suites.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

from common import (
    SEAL_VECTOR,
    build_engine,
    load_examples,
    nonnegative_int,
    positive_int,
    reset_prefix_cache,
    warmup,
)

MODES = ("eager", "split", "in_graph")


def distinct_config_specs(count: int, layers: list[int]):
    """Build distinct fingerprints without changing the selected tokens.

    Each payload differs in one value, while scale zero makes every config
    numerically inert.  Keeping the apply clause identical avoids coupling the
    K sweep to prompt-length-sensitive selectors.
    """
    from vllm.model_hooks.steering.loading import load_file_payload
    from vllm.model_hooks.steering.payloads import DirectionVector
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    base = load_file_payload(SEAL_VECTOR, format="gguf")
    vectors = {
        layer: np.array(value, copy=True) for layer, value in base.layers.items()
    }
    first_layer = min(vectors)
    specs = []
    for index in range(count):
        payload_vectors = {layer: value.copy() for layer, value in vectors.items()}
        payload_vectors[first_layer][0] += index * 1e-4
        specs.append(
            SteeringSpec(
                vectors=[
                    VectorSpec(
                        data=DirectionVector(payload_vectors),
                        scale=0.0,
                        layers=layers,
                        apply=ApplySpec(prompt="all", generation="all"),
                    )
                ]
            )
        )
    return specs


def run_mode(args):
    from vllm import SamplingParams

    llm = build_engine(
        args.mode, args.max_steer, max_num_seqs=args.max_num_seqs
    )
    params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
        ignore_eos=True,
    )
    probe = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    prompts = load_examples(args.batch)
    layers = list(range(args.layers))
    ks = sorted(set(args.configs))
    specs = distinct_config_specs(max(ks) or 1, layers)

    for k in ks:
        # A single shared spec must use the scalar API path.  Passing a list
        # of 512 identical specs needlessly exercises per-request admission
        # and does not represent one shared configuration.
        steering = (
            None
            if k == 0
            else specs[0]
            if k == 1
            else [specs[i % k] for i in range(args.batch)]
        )
        warmup(llm, prompts, steering)
        llm.generate(prompts, probe, steering=steering, use_tqdm=False)
        reset_prefix_cache(llm)
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
        default=None,
        help=(
            "max_steer_vectors override (default: vLLM resolves "
            "min(256, max_num_seqs))"
        ),
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
    if args.max_steer is not None and max(ks) > min(args.max_steer, args.batch):
        parser.error("K must not exceed --max-steer or --batch")
    if args.max_steer is None and max(ks) > args.batch:
        parser.error("K must not exceed --batch")

    if args.mode:
        run_mode(args)
        return

    passthrough = [
        "--batch",
        str(args.batch),
        *(
            ["--max-steer", str(args.max_steer)]
            if args.max_steer is not None
            else []
        ),
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
        for k in ks:
            print(f"===== tier: {mode}, K={k} =====", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--mode",
                    mode,
                    "--configs",
                    str(k),
                    *passthrough,
                ],
                check=True,
                env=env,
            )


if __name__ == "__main__":
    main()
