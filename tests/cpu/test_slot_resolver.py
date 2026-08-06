# SPDX-License-Identifier: Apache-2.0
"""Differential units for the runner's host-side slot resolver.

`resolve_slot_positions` re-implements the apply-spec semantics in
numpy so a step's trigger resolution costs one host pass regardless of
how many distinct configurations are live. The torch collector
(`collect_positions_apply_spec`, still serving capture) is the
reference implementation: every clause shape must resolve to exactly
the reference positions restricted to the slot's own tokens, on a
mixed continuous-batching step (chunked prefill, fresh prefill, decode
steps at different generation indices, unsteered co-batched requests).
"""

import numpy as np
import torch

from vllm.forward_context import BatchGeometry
from vllm.steer_vectors.api import ApplySpec
from vllm.steer_vectors.algorithms.clause import (
    clause_cache_key,
    collect_positions_apply_spec,
)
from vllm.v1.worker.gpu.steer_vector_utils import resolve_slot_positions

# One mixed step: [num_computed, num_prompt, num_output, seg_len].
# Requests 0/1/4 are prefilling (middle chunk, fresh, final chunk);
# 2/3/5 are decode steps at generation indices 0, 3 and 1.
REQS = [
    (8, 16, 0, 4),
    (0, 6, 0, 6),
    (5, 5, 1, 1),
    (10, 7, 4, 1),
    (12, 14, 0, 2),
    (4, 3, 2, 1),
]
# Request -> routing slot (-1 = unsteered; slot 1 serves two requests).
REQ_SLOTS = [0, 1, 1, 2, -1, 3]
TOKEN_IDS = [11, 7, 13, 14, 11, 12, 7, 14, 15, 16, 99, 7, 11, 7, 99]

CLAUSES = {
    0: [
        ApplySpec(prompt="all", generation="all").to_wire(),
        ApplySpec(prompt_window=(-9, -5)).to_wire(),
    ],
    1: [
        ApplySpec(prompt_positions=[-1, 0]).to_wire(),
        ApplySpec(
            prompt_window=(2, None),
            exclude_prompt_window=(3, 4),
        ).to_wire(),
    ],
    2: [
        ApplySpec(
            generation_window=(2, 5)
        ).to_wire(),
        ApplySpec(
            generation_positions=[3]
        ).to_wire(),
    ],
    3: [
        ApplySpec(
            generation_window=(0, 1)
        ).to_wire(),
        ApplySpec(
            prompt="all", generation="all",
            exclude_prompt_tokens=[7],
            exclude_generation_tokens=[7],
        ).to_wire(),
        ApplySpec(
            prompt_tokens=[11], exclude_prompt_positions=[0]
        ).to_wire(),
        # Union semantics: an unmatched token filter must not veto the
        # generation window (and vice versa).
        ApplySpec(
            generation_tokens=[123],
            generation_window=(0, 2),
        ).to_wire(),
        ApplySpec(
            generation="all", exclude_generation_positions=[1]
        ).to_wire(),
    ],
}


def make_geometry():
    seg_lens = np.array([r[3] for r in REQS], dtype=np.int32)
    qsl = np.concatenate(([0], np.cumsum(seg_lens))).astype(np.int32)
    assert int(qsl[-1]) == len(TOKEN_IDS)
    return BatchGeometry(
        query_start_loc=torch.from_numpy(qsl.astype(np.int64)),
        num_computed=torch.tensor([r[0] for r in REQS], dtype=torch.int32),
        num_prompt=torch.tensor([r[1] for r in REQS], dtype=torch.int32),
        num_output=torch.tensor([r[2] for r in REQS], dtype=torch.int32),
        req_ids=[f"req{i}" for i in range(len(REQS))],
        token_ids=torch.tensor(TOKEN_IDS, dtype=torch.int32),
        query_start_loc_cpu=qsl,
    )


def reference_positions(geo, token_slots_np, slot, clause):
    """Torch-collector positions restricted to the slot's tokens."""
    out = collect_positions_apply_spec(geo.token_ids, geo.samples_info(), clause)
    if out is None:
        return []
    return [p for p in out.tolist() if token_slots_np[p] == slot]


def test_resolver_matches_torch_collector_per_clause():
    geo = make_geometry()
    seg_lens = np.array([r[3] for r in REQS], dtype=np.int64)
    token_slots_np = np.repeat(np.array(REQ_SLOTS, dtype=np.int32), seg_lens)
    active_slots = sorted({s for s in REQ_SLOTS if s >= 0})

    resolved = resolve_slot_positions(
        CLAUSES, active_slots, token_slots_np, torch.device("cpu"), geo
    )

    checked = 0
    for slot, clauses in CLAUSES.items():
        for clause in clauses:
            key = clause_cache_key(clause)
            expected = reference_positions(geo, token_slots_np, slot, clause)
            actual = resolved[(slot, key)]
            if not expected:
                assert actual is None, (slot, clause, actual)
            else:
                assert actual is not None, (slot, clause, expected)
                assert actual.tolist() == expected, (slot, clause)
            checked += 1
    assert checked == 11
    # Sanity on the scenario itself: every clause family actually fired
    # somewhere (an all-None table would vacuously pass).
    fired = [k for k, v in resolved.items() if v is not None]
    assert len(fired) >= 4


def test_unsteered_requests_never_resolve():
    geo = make_geometry()
    seg_lens = np.array([r[3] for r in REQS], dtype=np.int64)
    token_slots_np = np.repeat(np.array(REQ_SLOTS, dtype=np.int32), seg_lens)
    active_slots = sorted({s for s in REQ_SLOTS if s >= 0})

    resolved = resolve_slot_positions(
        CLAUSES, active_slots, token_slots_np, torch.device("cpu"), geo
    )
    unsteered = {
        int(p)
        for p in np.nonzero(token_slots_np < 0)[0]
    }
    for positions in resolved.values():
        if positions is None:
            continue
        assert not (set(positions.tolist()) & unsteered)
