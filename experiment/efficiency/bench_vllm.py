# SPDX-License-Identifier: Apache-2.0
"""EasySteer/vLLM efficiency benchmark (EasySteer paper, Section 5.1).

Steering configurations, all with zero-valued vectors so the generated
text matches the baseline exactly:
    baseline      - no steering
    single_layer  - one vector at one layer (20)
    all_layer     - one vector on all 28 layers
    multi_vector  - three sequential vectors on all 28 layers

Sequential mode times 10 single-prompt requests; --batch N submits N
prompts in one generate call (vLLM continuous batching). --max-tokens
matches the paper's two settings (128 and 2048). Metrics: FTL (from a
timed 1-token request), TPS, TTLT.
"""

import argparse
import time

from common import MODEL, N_SEQUENTIAL, SEAL_VECTOR, load_examples, report

from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec


def zero_scale_spec(n_vectors, layers):
    return SteeringSpec(
        conflict="sequential",
        vectors=[
            VectorSpec(
                source=SEAL_VECTOR,
                scale=0.0,
                layers=layers,
                apply=ApplySpec(phases=["prompt", "generation"]),
            )
            for _ in range(n_vectors)
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=["baseline", "single_layer", "all_layer",
                                 "multi_vector"],
                        default="baseline")
    parser.add_argument("--batch", type=int, default=0,
                        help="batch size; 0 = sequential single requests")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        choices=[128, 2048])
    parser.add_argument("--cudagraph", action="store_true",
                        help="enable CUDA graphs (paper numbers are eager)")
    parser.add_argument("--graph-mode", choices=["split", "in_graph"],
                        default=None,
                        help="steering graph tier under --cudagraph: "
                             "full captures the steering kernel into the "
                             "graph (engine default when compiled); "
                             "piecewise splits at steered layers "
                             "(all algorithms)")
    args = parser.parse_args()

    steering = {
        "baseline": None,
        "single_layer": zero_scale_spec(1, [20]),
        "all_layer": zero_scale_spec(1, list(range(28))),
        "multi_vector": zero_scale_spec(3, list(range(28))),
    }[args.mode]
    engine_kwargs = {}
    if args.graph_mode is not None:
        engine_kwargs["steer_graph_mode"] = args.graph_mode
    llm = LLM(model=MODEL, enable_steer_vector=True,
              steer_algorithms=["direct"],
              steer_multi_vector=args.mode == "multi_vector",
              enforce_eager=not args.cudagraph, **engine_kwargs)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens,
                            skip_special_tokens=False)
    one_token = SamplingParams(temperature=0, max_tokens=1)

    if args.batch:
        examples = load_examples(args.batch)
        start = time.time()
        llm.generate(examples, one_token, steering=steering, use_tqdm=False)
        ftl = time.time() - start
        start = time.time()
        outs = llm.generate(examples, params, steering=steering)
        elapsed = time.time() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        report(tokens, elapsed, args.batch, ftl_s=ftl)
    else:
        examples = load_examples(N_SEQUENTIAL)
        start = time.time()
        llm.generate(examples[0], one_token, steering=steering, use_tqdm=False)
        ftl = time.time() - start
        tokens = 0
        start = time.time()
        for example in examples:
            outs = llm.generate(example, params, steering=steering,
                                use_tqdm=False)
            tokens += len(outs[0].outputs[0].token_ids)
        elapsed = time.time() - start
        report(tokens, elapsed, len(examples), ftl_s=ftl)


if __name__ == "__main__":
    main()
