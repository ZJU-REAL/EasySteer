# SPDX-License-Identifier: Apache-2.0
"""Worker stores deduplicate immutable content without touching source files."""

import types
from unittest import mock

import numpy as np
import pytest
import torch
from vllm.model_hooks.steering.payload_cache import PayloadCache
from vllm.model_hooks.steering.payloads import (
    DirectionVector,
    LinearMap,
    LowRankProjector,
    ReftIntervention,
    RouterConfig,
    materialize,
)


def store(capacity=8):
    config = types.SimpleNamespace(
        max_steer_vectors=capacity, adapter_dtype=torch.float32
    )
    return PayloadCache("cpu", config)


def test_content_reuse_performs_no_file_io():
    cache = store()
    wire = DirectionVector({3: np.ones(4)}).to_wire()
    with mock.patch("os.stat", side_effect=AssertionError("worker touched filesystem")):
        first = cache.get(wire)
        repeated = cache.get(DirectionVector({3: np.ones(4)}).to_wire())
        changed = cache.get(DirectionVector({3: np.full(4, 2.0)}).to_wire())
    assert first is repeated
    assert changed is not first
    assert torch.equal(first[3], torch.ones(4))
    assert torch.equal(changed[3], torch.full((4,), 2.0))


@pytest.mark.parametrize("payload", [
    LinearMap(np.eye(8), np.ones(8)),
    LowRankProjector(np.ones((8, 2)), np.ones((8, 2))),
    ReftIntervention(np.ones((8, 2)), np.ones((2, 8)), np.ones(2)),
], ids=["linear", "lowrank", "reft"])
def test_broadcast_payload_materializes_once_across_target_layers(payload):
    cache = store(capacity=1)
    wire = payload.to_wire()
    with mock.patch(
        "vllm.model_hooks.steering.payload_cache.materialize", wraps=materialize
    ) as allocate:
        cache.preload(wire, target_layers=[11])
        first = cache.get(wire, target_layers=[11])
        other = cache.get(wire, target_layers=[3, 5])
        reordered = cache.get(wire, target_layers=[5, 3])
        assert allocate.call_count == 1
    assert set(first) == {11}
    assert set(other) == {3, 5}
    assert len(cache._entries) == 1
    for name, tensor in first[11].items():
        assert tensor is other[3][name] is other[5][name] is reordered[5][name]
    # Layer mappings stay private, while their immutable tensor storage is shared.
    first[11].clear()
    assert other[3]
    assert cache.get(wire, target_layers=[11])[11]
    with pytest.raises(ValueError, match="layers is required"):
        cache.get(wire)


def test_lru_eviction_preserves_active_payload():
    cache = store(capacity=1)
    first_wire = DirectionVector({3: np.ones(4)}).to_wire()
    second_wire = DirectionVector({3: np.full(4, 2.0)}).to_wire()
    first = cache.get(first_wire)
    cache.preload(second_wire)
    assert len(cache._entries) == 1
    assert torch.equal(first[3], torch.ones(4))
    assert cache.get(first_wire) is not first


def test_router_uses_same_materialization_and_deduplication():
    cache = store()
    payload = RouterConfig({5: {"mode": "soft_topk", "expert_ids": [1], "topk": 2}})
    first = cache.get(payload.to_wire())
    repeated = cache.get(payload.to_wire())
    assert first is repeated
    assert first == payload.layers


def test_preloaded_layer_keyed_payloads_share_one_entry_across_target_subsets():
    cache = store()
    for payload in (
        DirectionVector({1: np.ones(4), 2: np.ones(4)}),
        RouterConfig({1: {"expert_ids": [0]}, 2: {"expert_ids": [1]}}),
    ):
        wire = payload.to_wire()
        cache.preload(wire)
        complete = cache.get(wire)
        assert cache.get(wire, target_layers=[1]) is complete
        assert cache.get(wire, target_layers=[2]) is complete
        assert set(complete) == {1, 2}


def test_slots_scale_private_payloads_without_mutating_cached_tensors():
    from vllm.model_hooks.steering.controllers import HiddenStatesController

    cache = store()
    wire = DirectionVector({0: np.ones(4)}).to_wire()
    source = cache.get(wire)[0]
    controller = HiddenStatesController()
    for slot, scale in enumerate((2.0, -1.0)):
        controller.configure_slot(slot, [
            {"algorithm": "direct", "payload": source, "scale": scale},
        ])
        torch.testing.assert_close(
            controller.slot_interventions[slot][0].payload,
            torch.full((4,), scale),
        )
    controller.clear_slot(0)
    assert torch.equal(source, torch.ones(4))
    assert cache.get(wire)[0] is source
    assert torch.equal(controller.slot_interventions[1][0].payload, -torch.ones(4))
