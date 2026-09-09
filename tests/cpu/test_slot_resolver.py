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

from types import SimpleNamespace

import numpy as np
import torch
from vllm.model_hooks.selection.batch import BatchGeometry
from vllm.model_hooks.selection.runtime import (
    clause_cache_key,
    collect_positions_apply_spec,
)
from vllm.model_hooks.steering.api import ApplySpec
from vllm.v1.worker.gpu.model_hook_utils import build_batch_geometry
from vllm.v1.worker.gpu.steer_vector_utils import (
    fill_graph_steer_buffers,
    make_steering_forward_kwargs,
    resolve_slot_positions,
)

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
        ApplySpec(generation_window=(2, 5)).to_wire(),
        ApplySpec(generation_positions=[3]).to_wire(),
    ],
    3: [
        ApplySpec(generation_window=(0, 1)).to_wire(),
        ApplySpec(
            prompt="all",
            generation="all",
            exclude_prompt_tokens=[7],
            exclude_generation_tokens=[7],
        ).to_wire(),
        ApplySpec(prompt_tokens=[11], exclude_prompt_positions=[0]).to_wire(),
        # Union semantics: an unmatched token filter must not veto the
        # generation window (and vice versa).
        ApplySpec(
            generation_tokens=[123],
            generation_window=(0, 2),
        ).to_wire(),
        ApplySpec(generation="all", exclude_generation_positions=[1]).to_wire(),
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
    out = collect_positions_apply_spec(geo.token_ids, geo.device_view(), clause)
    if out is None:
        return []
    return [p for p in out.tolist() if token_slots_np[p] == slot]


def test_resolver_matches_torch_collector_per_clause():
    geo = make_geometry()
    seg_lens = np.array([r[3] for r in REQS], dtype=np.int64)
    token_slots_np = np.repeat(np.array(REQ_SLOTS, dtype=np.int32), seg_lens)
    active_slots = sorted({s for s in REQ_SLOTS if s >= 0})

    resolved = resolve_slot_positions(
        CLAUSES, active_slots, np.asarray(REQ_SLOTS), torch.device("cpu"), geo
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


def test_recomputed_outputs_keep_original_prompt_boundary_and_decode_indices():
    """Re-prefill spans both phases; ordinary speculative decode keeps step indices."""
    from vllm.model_hooks.capture.selection import may_select_step_rows

    offsets = np.array([0, 7, 10], dtype=np.int32)
    batch = SimpleNamespace(
        req_ids=["resumed", "decode"], num_reqs=2, num_tokens=10,
        query_start_loc_np=offsets, query_start_loc=torch.from_numpy(offsets),
        num_computed_tokens_np=np.array([2, 7], dtype=np.int32),
        prefill_len_np=np.array([9, 4], dtype=np.int32),
        is_prefilling_np=np.array([True, False]),
        input_ids=torch.arange(10, dtype=torch.int32),
    )
    geometry = build_batch_geometry(batch, np.array([4, 4], dtype=np.int32))
    assert geometry.num_prompt.tolist() == [4, 4]
    assert geometry.num_output.tolist() == [0, 4]
    cases = [
        (ApplySpec(prompt_positions=[-1]), [1]),
        (ApplySpec(generation_positions=[1]), [3]),
        (ApplySpec(generation_positions=[3]), [5, 7, 8, 9]),
    ]
    clauses = [spec.to_wire() for spec, _ in cases]
    positions = resolve_slot_positions(
        {0: clauses}, [0], np.array([0, 0]), torch.device("cpu"), geometry,
    )
    for clause, (_, expected) in zip(clauses, cases):
        assert positions[(0, clause_cache_key(clause))].tolist() == expected
        actual = collect_positions_apply_spec(
            geometry.token_ids, geometry.device_view(), clause,
        )
        assert actual.tolist() == expected
        assert may_select_step_rows(clause, 2, 7, 4, True)
    assert not may_select_step_rows(
        ApplySpec(generation_positions=[8]).to_wire(), 2, 7, 4, True,
    )


def test_reallocated_drafts_route_by_actual_rows_after_request_reordering():
    """GPU draft redistribution must not assign another request's rows to a slot.

    The scheduled budget and even CPU layout both differ from the actual
    forward layout. CPU tensors stand in for the GPU-produced offsets so this
    checks both eager plans and graph buffers without booting an engine.
    """
    actual_offsets = np.array([0, 1, 2, 5], dtype=np.int32)
    batch = SimpleNamespace(
        req_ids=["second", "plain", "first"],
        num_reqs=3,
        num_tokens=5,
        num_tokens_after_padding=8,
        num_scheduled_tokens=np.array([4, 1, 4], dtype=np.int32),
        query_start_loc_np=np.array([0, 2, 3, 5], dtype=np.int32),
        query_start_loc=torch.from_numpy(actual_offsets),
        num_computed_tokens_np=np.array([8, 7, 9], dtype=np.int32),
        prefill_len_np=np.array([5, 5, 5], dtype=np.int32),
        is_prefilling_np=np.zeros(3, dtype=bool),
        input_ids=torch.tensor([20, 30, 10, 11, 12], dtype=torch.int32),
    )
    slots = {"first": 0, "second": 1}
    clause = ApplySpec(generation="all").to_wire()
    group = ("priority", (clause_cache_key(clause),))
    masks = torch.zeros(2, 8)
    controllers = [SimpleNamespace(graph_mask=masks[i]) for i in range(2)]
    manager = SimpleNamespace(
        slot_for_request=slots.get,
        slot_clauses=lambda: {0: [clause], 1: [clause]},
        slot_groups=lambda: {0: (group,), 1: (group,)},
        token_rows_buf=torch.full((8,), 99, dtype=torch.int64),
        graph_masks_buf=masks,
        zero_graph_masks=lambda n: [c.graph_mask[:n].zero_() for c in controllers],
        graph_batch_entries=lambda: {
            slot: (
                slot + 1,
                SimpleNamespace(
                    vectors=[SimpleNamespace(apply_spec=clause, algorithm="direct")]
                ),
                [controllers[slot]],
            )
            for slot in range(2)
        },
    )
    geometry = build_batch_geometry(
        batch, batch.prefill_len_np, query_start_loc_cpu=actual_offsets
    )
    kwargs = make_steering_forward_kwargs(
        batch,
        manager=manager,
        geometry=geometry,
    )
    assert kwargs["steer_active_slots"] == [0, 1]
    assert kwargs["steer_slot_positions"][(0, group, 0)].tolist() == [2, 3, 4]
    assert kwargs["steer_slot_positions"][(1, group, 0)].tolist() == [0]

    fill_graph_steer_buffers(batch, manager, geometry=geometry)
    assert manager.token_rows_buf.tolist() == [2, 0, 1, 1, 1, 0, 0, 0]
    assert controllers[0].graph_mask.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert controllers[1].graph_mask.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]

    # An idle small graph need only clear its own span. Replaying a larger
    # bucket later must clear the old steering rows before they become padding.
    manager.graph_batch_entries = dict
    batch.num_tokens_after_padding = 2
    fill_graph_steer_buffers(batch, manager)
    assert manager.token_rows_buf[:2].tolist() == [0, 0]
    batch.num_tokens_after_padding = 8
    fill_graph_steer_buffers(batch, manager)
    assert manager.token_rows_buf.tolist() == [0] * 8
    assert all(c.graph_mask.tolist() == [0] * 8 for c in controllers)


def test_empty_request_segments_never_claim_another_requests_rows():
    """Empty leading/trailing segments neither alias a neighbor nor index past it."""
    offsets = np.array([0, 0, 2, 2], dtype=np.int32)
    batch = SimpleNamespace(
        req_ids=["empty_head", "filled", "empty_tail"],
        num_reqs=3,
        num_tokens=2,
        num_scheduled_tokens=np.array([0, 2, 0], dtype=np.int32),
        query_start_loc_np=offsets,
        query_start_loc=torch.from_numpy(offsets),
        num_computed_tokens_np=np.array([5, 5, 5], dtype=np.int32),
        prefill_len_np=np.array([3, 3, 3], dtype=np.int32),
        is_prefilling_np=np.zeros(3, dtype=bool),
        input_ids=torch.tensor([11, 12], dtype=torch.int32),
    )
    slots = dict(zip(batch.req_ids, range(3)))
    clause = ApplySpec(generation="all").to_wire()
    group = ("priority", (clause_cache_key(clause),))
    manager = SimpleNamespace(
        slot_for_request=slots.get,
        slot_clauses=lambda: {s: [clause] for s in range(3)},
        slot_groups=lambda: {s: (group,) for s in range(3)},
    )
    kwargs = make_steering_forward_kwargs(
        batch, manager=manager, geometry=build_batch_geometry(batch, batch.prefill_len_np)
    )
    assert kwargs["steer_active_slots"] == [0, 1, 2]
    assert kwargs["steer_slot_positions"][(0, group, 0)] is None
    assert kwargs["steer_slot_positions"][(1, group, 0)].tolist() == [0, 1]
    assert kwargs["steer_slot_positions"][(2, group, 0)] is None


def test_graph_scatter_combines_slots_without_crossing_mask_families():
    """Two direct slots share one mask; replacement uses the same layer's other mask."""
    offsets = np.array([0, 2, 3, 5, 6], dtype=np.int32)
    batch = SimpleNamespace(
        req_ids=["direct-a", "direct-b", "replace", "plain"],
        num_reqs=4,
        num_tokens=6,
        num_tokens_after_padding=8,
        query_start_loc_np=offsets,
        query_start_loc=torch.from_numpy(offsets),
        num_computed_tokens_np=np.full(4, 3, dtype=np.int32),
        prefill_len_np=np.full(4, 3, dtype=np.int32),
        is_prefilling_np=np.zeros(4, dtype=bool),
        input_ids=torch.arange(6, dtype=torch.int32),
    )
    slots = {"direct-a": 0, "direct-b": 1, "replace": 2}
    clause = ApplySpec(generation="all").to_wire()
    masks = torch.full((4, 8), 99.0)
    controller = SimpleNamespace(graph_mask=masks[2], replace_mask=masks[3])
    # Nonzero storage offsets ensure scatter writes into each actual mask.
    entries = {
        slot: (
            row,
            SimpleNamespace(
                vectors=[SimpleNamespace(apply_spec=clause, algorithm=algorithm)]
            ),
            [controller],
        )
        for slot, row, algorithm in (
            (0, 1, "direct"), (1, 2, "direct"), (2, 3, "replace")
        )
    }
    manager = SimpleNamespace(
        slot_for_request=slots.get,
        slot_clauses=lambda: {slot: [clause] for slot in entries},
        token_rows_buf=torch.full((8,), 99, dtype=torch.int64),
        graph_masks_buf=masks,
        zero_graph_masks=lambda n: masks[:, :n].zero_(),
        graph_batch_entries=lambda: entries,
    )

    fill_graph_steer_buffers(
        batch, manager, geometry=build_batch_geometry(batch, batch.prefill_len_np)
    )

    assert manager.token_rows_buf.tolist() == [1, 1, 2, 3, 3, 0, 0, 0]
    assert masks[:2].count_nonzero() == 0
    assert controller.graph_mask.tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert controller.replace_mask.tolist() == [0, 0, 0, 1, 1, 0, 0, 0]


def _request_with_layer_clauses(layer_clauses, *, conflict="priority"):
    from vllm.model_hooks.steering.api import to_engine_request
    from vllm.steer_vectors import DirectionVector, SteeringSpec, VectorSpec

    return to_engine_request(SteeringSpec(
        vectors=[
            VectorSpec(
                data=DirectionVector({layer: np.ones(4) for layer in layers}),
                apply=clause,
            )
            for layers, clause in layer_clauses
        ],
        conflict=conflict,
    ))


def test_conflict_rejects_only_affected_request_in_a_shared_slot():
    from vllm.model_hooks.steering.validation import request_position_groups

    request = _request_with_layer_clauses([
        ([0], ApplySpec(prompt="all", generation="all")),
        ([0], ApplySpec(prompt_tokens=[7], generation_tokens=[7])),
    ], conflict="error")
    groups = request_position_groups(request)
    errors = {}
    # The fresh prefill and the decode request share a configuration. Only
    # prefill contains token 7, so overlap must leave its peer's steering intact.
    resolved = resolve_slot_positions(
        {0: [vector.apply_spec for vector in request.vectors]},
        [0], np.array([-1, 0, 0, -1, -1, -1]), torch.device("cpu"), make_geometry(),
        slot_groups={0: groups}, errors=errors,
    )
    assert set(errors) == {"req1"}
    assert resolved[(0, groups[0], 0)].tolist() == [10]
    assert resolved[(0, groups[0], 1)] is None


def test_priority_on_one_layer_does_not_claim_another_layers_positions():
    from vllm.model_hooks.steering.validation import request_position_groups

    request = _request_with_layer_clauses([
        ([0], ApplySpec(prompt="all", generation="all")),
        ([1], ApplySpec(generation="all")),
    ])
    groups = request_position_groups(request)
    assert len(groups) == 2
    resolved = resolve_slot_positions(
        {0: [vector.apply_spec for vector in request.vectors]},
        [0], np.array([-1, 0, 0, -1, -1, -1]), torch.device("cpu"), make_geometry(),
        slot_groups={0: groups},
    )
    assert resolved[(0, groups[0], 0)].tolist() == list(range(4, 11))
    assert resolved[(0, groups[1], 0)].tolist() == [10]


def test_identical_ordered_interventions_share_one_resolution_across_layers():
    from vllm.model_hooks.steering.validation import request_position_groups

    request = _request_with_layer_clauses([
        ([0, 2, 4], ApplySpec(prompt_tokens=[7], generation_tokens=[7])),
        ([0, 2, 4], ApplySpec(prompt="all", generation="all")),
    ])
    groups = request_position_groups(request)
    assert len(groups) == 1
    resolved = resolve_slot_positions(
        {0: [vector.apply_spec for vector in request.vectors]},
        [0], np.array([-1, 0, 0, -1, -1, -1]), torch.device("cpu"), make_geometry(),
        slot_groups={0: groups},
    )
    assert len(resolved) == 2
    assert resolved[(0, groups[0], 0)].tolist() == [6]
    assert resolved[(0, groups[0], 1)].tolist() == [4, 5, 7, 8, 9, 10]
