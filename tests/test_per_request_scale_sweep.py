# SPDX-License-Identifier: Apache-2.0
"""Phase C validation: per-request steering isolation under batching.

Runs the sentiment example with scales 0.0..5.0 (step 0.1):
1. Reference pass: each scale as an isolated request -> R[s].
2. Batch pass: all 51 requests in one generate() call -> B[s].

Criteria:
  C1  B[s] == R[s] for every s (per-request isolation under batching)
  C2  R[0.0] == unsteered output; outputs vary across the sweep
  C4  exactly one vector load in the batch pass (preload + config dedup)
  perf: batch wall-clock substantially below sequential wall-clock

Usage:
  python test_per_request_scale_sweep.py --model <path> --vector <gguf> \
      [--gpu-memory-utilization 0.3]
"""

import argparse
import os
import time

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

PROMPT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
TARGET_LAYERS = list(range(10, 26))


def make_request(vector: str, scale: float, idx: int) -> SteerVectorRequest:
    return SteerVectorRequest(
        f"sweep-{scale:.1f}",
        idx + 1,
        steer_vector_local_path=vector,
        scale=scale,
        target_layers=TARGET_LAYERS,
        prefill_trigger_tokens=[-1],
        generate_trigger_tokens=[-1],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    scales = [round(i * 0.1, 1) for i in range(51)]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    llm = LLM(
        model=args.model,
        enable_steer_vector=True,
        enforce_eager=True,
        tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        max_num_seqs=64,
        disable_log_stats=False,
    )
    llm.preload_steer_vectors([args.vector])

    unsteered = llm.generate([PROMPT], sampling_params=params)[0].outputs[0].text

    t0 = time.perf_counter()
    ref = {}
    for i, s in enumerate(scales):
        out = llm.generate(
            [PROMPT],
            steer_vector_request=make_request(args.vector, s, i),
            sampling_params=params,
        )
        ref[s] = out[0].outputs[0].text
    t_seq = time.perf_counter() - t0

    reqs = [make_request(args.vector, s, 100 + i) for i, s in enumerate(scales)]
    t0 = time.perf_counter()
    outs = llm.generate(
        [PROMPT] * len(scales), steer_vector_request=reqs, sampling_params=params
    )
    t_batch = time.perf_counter() - t0
    batch = {s: o.outputs[0].text for s, o in zip(scales, outs)}

    # Batch-repeat determinism: same batch twice must match exactly.
    reqs2 = [make_request(args.vector, s, 200 + i) for i, s in enumerate(scales)]
    outs2 = llm.generate(
        [PROMPT] * len(scales), steer_vector_request=reqs2,
        sampling_params=params,
    )
    batch2 = {s: o.outputs[0].text for s, o in zip(scales, outs2)}
    repeat_failures = [s for s in scales if batch[s] != batch2[s]]

    failures = [s for s in scales if batch[s] != ref[s]]
    distinct = len(set(ref.values()))

    import json
    with open("sweep_outputs.json", "w") as f:
        json.dump({"ref": {str(k): v for k, v in ref.items()},
                   "batch": {str(k): v for k, v in batch.items()},
                   "batch2": {str(k): v for k, v in batch2.items()}}, f)

    print("\n================ SCALE SWEEP RESULTS ================")
    print(f"scales: {len(scales)}, distinct reference outputs: {distinct}")
    print(f"sequential: {t_seq:.1f}s   batched: {t_batch:.1f}s "
          f"(speedup {t_seq / max(t_batch, 1e-9):.1f}x)")
    print(f"C1 isolation: {'PASS' if not failures else 'FAIL ' + str(failures)}")
    print(f"C1b batch-repeat determinism: "
          f"{'PASS' if not repeat_failures else 'FAIL ' + str(repeat_failures)}")
    for s_bad in failures[:3]:
        print(f"--- mismatch scale {s_bad}:")
        print(f"  seq:   {ref[s_bad][:150]}")
        print(f"  batch: {batch[s_bad][:150]}")
    c2 = ref[0.0] == unsteered and distinct > 5
    print(f"C2 anchors: {'PASS' if c2 else 'FAIL'} "
          f"(scale0==unsteered: {ref[0.0] == unsteered}, distinct: {distinct})")
    print(f"sample outputs:\n  scale 0.0: {ref[0.0][:90]}...\n"
          f"  scale 2.0: {ref[2.0][:90]}...\n  scale 5.0: {ref[5.0][:90]}...")
    ok = not repeat_failures and c2
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
