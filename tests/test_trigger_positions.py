# SPDX-License-Identifier: Apache-2.0
"""Phase C validation: trigger positions under heterogeneous batching.

Batches requests whose configs differ in trigger type, trigger positions
and target layers, with steering trace enabled, then checks the recorded
apply events against an oracle computed from the trace's own batch
records:

  - prefill_trigger_positions: only those absolute prompt positions,
    only in prefill phase
  - prefill_trigger_tokens: only positions holding those token ids
  - generate_first_k_tokens: only the first k decode steps
  - target_layers: apply records appear at exactly those layers
  - isolation: tokens of the unsteered request never appear in any
    apply record

Usage:
  VLLM_STEER_TRACE_DIR must NOT be preset; the script sets it.
  python test_trigger_positions.py --model <path> --vector <gguf>
"""

import argparse
import collections
import json
import os
import tempfile

TRACE_DIR = tempfile.mkdtemp(prefix="steer_trace_")
os.environ["VLLM_STEER_TRACE_DIR"] = TRACE_DIR

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.steer_vectors.request import SteerVectorRequest  # noqa: E402

PROMPT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        enable_steer_vector=True,
        enforce_eager=os.environ.get("STEER_TEST_EAGER", "1") == "1",
        tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        max_num_seqs=16,
    )
    llm.preload_steer_vectors([args.vector])

    tok = llm.get_tokenizer()
    prompt_ids = tok.encode(PROMPT)
    trigger_token = prompt_ids[5]

    configs = {
        # slot key -> (request, expectation descriptor)
        "pos01": SteerVectorRequest(
            "pos01", 1, steer_vector_local_path=args.vector, scale=1.0,
            target_layers=[10, 11, 12],
            prefill_trigger_positions=[0, 1],
        ),
        "token5": SteerVectorRequest(
            "token5", 2, steer_vector_local_path=args.vector, scale=1.0,
            target_layers=[13, 14],
            prefill_trigger_tokens=[trigger_token],
        ),
        "firstk3": SteerVectorRequest(
            "firstk3", 3, steer_vector_local_path=args.vector, scale=1.0,
            target_layers=[15],
            generate_first_k_tokens=3,
        ),
    }
    order = ["pos01", "token5", "firstk3", None]  # None = unsteered request
    reqs = [configs[k] if k else None for k in order]
    params = SamplingParams(temperature=0.0, max_tokens=16)
    llm.generate([PROMPT] * len(order), steer_vector_request=reqs,
                 sampling_params=params)

    # ---- parse trace ----
    steps, applies = {}, collections.defaultdict(list)
    for fn in os.listdir(TRACE_DIR):
        with open(os.path.join(TRACE_DIR, fn)) as f:
            for line in f:
                rec = json.loads(line)
                if rec["type"] == "step":
                    steps[rec["step"]] = rec
                else:
                    applies[rec["step"]].append(rec)

    # slot id by request name: recover from step records via request order
    # (req_id format is "<submission_index>-<uuid>"). Scan all steps since
    # prefills may be scheduled one per step.
    slot_by_key = {}
    for step in steps.values():
        for req_id, slot in zip(step["req_ids"], step["slots"]):
            if slot < 0:
                continue
            idx = int(str(req_id).split("-", 1)[0])
            if 0 <= idx < len(order) and order[idx]:
                slot_by_key[order[idx]] = slot
    expected_layers = {"pos01": {10, 11, 12}, "token5": {13, 14},
                       "firstk3": {15}}

    errors = []
    seen_layers = collections.defaultdict(set)
    for step_idx, step in steps.items():
        qsl = step["query_start_loc"]
        n = len(step["req_ids"])
        spans = {step["slots"][i]: (qsl[i], qsl[i + 1], i) for i in range(n)}

        expected = collections.defaultdict(set)
        for key in slot_by_key:
            slot = slot_by_key[key]
            if slot not in spans:
                continue
            start, end, i = spans[slot]
            num_computed = step["num_computed"][i]
            num_output = step["num_output"][i]
            is_prefill = num_output == 0
            for p in range(start, end):
                abs_pos = num_computed + (p - start)
                token = step["token_ids"][p]
                if key == "pos01" and is_prefill and abs_pos in (0, 1):
                    expected[slot].add(p)
                elif key == "token5" and is_prefill and token == trigger_token:
                    expected[slot].add(p)
                # Implementation semantics: gen_count < k, where the
                # runner reports count 1 at the first decode step (the
                # k-th output token is sampled from prefill logits).
                elif key == "firstk3" and not is_prefill and num_output < 3:
                    expected[slot].add(p)

        got = collections.defaultdict(dict)
        for rec in applies.get(step_idx, []):
            got[rec["slot"]][rec["layer"]] = set(rec["positions"])

        for key, slot in slot_by_key.items():
            layer_maps = got.get(slot, {})
            seen_layers[key].update(layer_maps)
            for layer, positions in layer_maps.items():
                if layer not in expected_layers[key]:
                    errors.append(
                        f"step {step_idx}: {key} applied at layer {layer} "
                        f"not in target_layers")
                if positions != expected[slot]:
                    errors.append(
                        f"step {step_idx} layer {layer}: {key} positions "
                        f"{sorted(positions)} != expected "
                        f"{sorted(expected[slot])}")
            if expected[slot] and not layer_maps:
                errors.append(
                    f"step {step_idx}: {key} expected "
                    f"{sorted(expected[slot])} but no apply records")

        # Unsteered request isolation: its span never appears anywhere.
        unsteered_spans = [
            (qsl[i], qsl[i + 1]) for i in range(n) if step["slots"][i] == -1
        ]
        for slot, layer_maps in got.items():
            for layer, positions in layer_maps.items():
                for s, e in unsteered_spans:
                    hit = [p for p in positions if s <= p < e]
                    if hit:
                        errors.append(
                            f"step {step_idx} layer {layer}: slot {slot} "
                            f"steered unsteered request tokens {hit}")

    for key, layers in seen_layers.items():
        missing = expected_layers[key] - layers
        if missing:
            errors.append(f"{key}: no apply records at layers {missing}")
    if len(slot_by_key) != 3:
        errors.append(f"expected 3 mapped configs, got {slot_by_key}")

    print("\n================ TRIGGER POSITION RESULTS ================")
    print(f"trace: {len(steps)} steps, slots: {slot_by_key}")
    for e in errors[:20]:
        print("ERROR:", e)
    print("OVERALL:", "PASS" if not errors else f"FAIL ({len(errors)} errors)")
    print(f"trace dir kept at: {TRACE_DIR}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
