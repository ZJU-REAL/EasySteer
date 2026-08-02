# SPDX-License-Identifier: Apache-2.0
"""Fuzz-check the mask-algebra position collector against the legacy one.

Runs thousands of randomized batch geometries and trigger configs (plus
directed edge cases) through both the current
collect_positions_gpu_batch and the verbatim pre-refactor copy in
_legacy_position_collector.py, asserting identical position sets.
CPU-only.
"""

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _legacy_position_collector as legacy  # noqa: E402

from vllm.steer_vectors.algorithms.parameter_control import (  # noqa: E402
    collect_positions_gpu_batch as new_collect,
)

VOCAB = list(range(1, 50))


def random_case(rng: random.Random):
    num_samples = rng.randint(1, 6)
    lengths = [
        1 if rng.random() < 0.5 else rng.randint(2, 12)
        for _ in range(num_samples)
    ]
    total = sum(lengths)
    qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths), 0)),
                       dtype=torch.long)
    is_decode = torch.tensor([ln == 1 for ln in lengths])
    num_computed = None
    if rng.random() < 0.5:
        num_computed = torch.tensor(
            [rng.randint(0, 8) for _ in range(num_samples)], dtype=torch.long
        )
    num_output = None
    if rng.random() < 0.7:
        num_output = torch.tensor(
            [rng.randint(0, 6) for _ in range(num_samples)],
            dtype=torch.int32,
        )
    tokens = torch.tensor([rng.choice(VOCAB) for _ in range(total)],
                          dtype=torch.long)

    def maybe_token_set(p_all=0.15):
        r = rng.random()
        if r < 0.35:
            return None
        if r < 0.35 + p_all:
            return {-1}
        return set(rng.sample(VOCAB, rng.randint(1, 5)))

    def maybe_positions():
        if rng.random() < 0.5:
            return None
        return [rng.randint(-5, 12) for _ in range(rng.randint(1, 4))]

    cfg = dict(
        prefill_trigger_tokens=maybe_token_set(),
        prefill_trigger_positions=maybe_positions(),
        prefill_exclude_tokens=maybe_token_set(p_all=0.0),
        prefill_exclude_positions=maybe_positions(),
        generate_trigger_tokens=maybe_token_set(),
        generate_first_k_tokens=None,
        generate_after_k_tokens=None,
    )
    which_k = rng.random()
    if which_k < 0.3:
        cfg["generate_first_k_tokens"] = rng.randint(0, 5)
    elif which_k < 0.6:
        cfg["generate_after_k_tokens"] = rng.randint(0, 5)
    has_prefill = (
        cfg["prefill_trigger_tokens"] is not None
        or cfg["prefill_trigger_positions"] is not None
    )
    samples_info = {
        "query_start_loc": qsl,
        "num_computed": num_computed,
        "is_decode_mask": is_decode,
        "num_output_tokens": num_output,
    }
    hidden = torch.zeros(total, 4)
    return hidden, tokens, samples_info, cfg, has_prefill


def run_both(hidden, tokens, samples_info, cfg, has_prefill):
    old = legacy.collect_positions_gpu_batch(
        hidden_states=hidden, current_tokens=tokens,
        samples_info=dict(samples_info), has_prefill_triggers=has_prefill,
        **cfg,
    )
    new = new_collect(
        hidden_states=hidden, current_tokens=tokens,
        samples_info=dict(samples_info), has_prefill_triggers=has_prefill,
        **cfg,
    )
    old_set = set() if old is None else set(old.tolist())
    new_set = set() if new is None else set(new.tolist())
    return old_set, new_set


def main() -> int:
    rng = random.Random(20260802)
    failures = 0
    n_cases = 5000
    for i in range(n_cases):
        case = random_case(rng)
        old_set, new_set = run_both(*case)
        if old_set != new_set:
            failures += 1
            if failures <= 3:
                _, _, samples_info, cfg, has_prefill = case
                print(f"MISMATCH case {i}: old={sorted(old_set)} "
                      f"new={sorted(new_set)}\n  cfg={cfg} "
                      f"has_prefill={has_prefill}\n  info={samples_info}")
    print(f"fuzz: {n_cases} cases, {failures} mismatches")
    print("OVERALL:", "PASS" if failures == 0 else "FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
