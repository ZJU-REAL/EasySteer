# SPDX-License-Identifier: Apache-2.0
"""Capture storage preserves labels, wire dtypes and reusable row budgets."""

import pytest
import torch
from vllm.model_hooks.capture.serialization import (
    assemble_captured,
    deserialize_captured,
)
from vllm.model_hooks.capture.session import CaptureSession
from vllm.model_hooks.capture.store import StreamConfig, StreamStore
from vllm.model_hooks.components.registry import COMPONENTS


def append_rows(store, layer=0, request="a", values=None, name="layer.0"):
    values = torch.arange(6.0).reshape(3, 2) if values is None else values
    rows = values.shape[0]
    meta = torch.stack(
        [
            torch.full((rows,), store.req_index(request), dtype=torch.int32),
            torch.arange(rows, dtype=torch.int32),
            torch.arange(10, 10 + rows, dtype=torch.int32),
        ],
        dim=1,
    )
    store.append(layer, values, meta, name)
    store.flush()
    return values


@pytest.mark.parametrize(
    "dtype", ["float16", "bfloat16", "float32", "float64", "int32", "int64"]
)
def test_every_accepted_storage_dtype_roundtrips_without_upcasting(dtype):
    store = StreamStore(StreamConfig(dtype=dtype, budget_rows=3))
    values = append_rows(store)
    raw = store.serialize()
    tensors, labels = deserialize_captured(raw)
    expected = values.to(getattr(torch, dtype))
    assert tensors[0].dtype == expected.dtype
    assert len(raw[0]["data"]) == expected.numel() * expected.element_size()
    torch.testing.assert_close(tensors[0], expected)
    assert labels[0].req_ids == ["a"] * 3
    assert labels[0].positions.tolist() == [0, 1, 2]
    assert labels[0].token_ids.tolist() == [10, 11, 12]


def test_replicated_capture_width_metadata_does_not_require_attention_heads():
    store = StreamStore(StreamConfig())
    values = append_rows(store)
    raw = store.serialize()
    raw[0]["layout"] = {"width": 2}
    raw[0]["shard"] = {
        "kind": "replicated",
        "tp_rank": 0,
        "tp_size": 2,
        "feature_start": 0,
        "global_width": 2,
    }
    tensors, labels, layouts = assemble_captured([{}, raw], tp_size=2)
    torch.testing.assert_close(tensors[0], values)
    assert labels[0].positions.tolist() == [0, 1, 2]
    assert layouts == {0: {"width": 2}}


@pytest.mark.parametrize("dtype", ["bool", "int16", "complex64", "missing"])
def test_unsupported_wire_dtype_is_rejected_when_enabling_stream(dtype):
    with pytest.raises(ValueError, match="unsupported capture storage dtype"):
        StreamConfig(dtype=dtype)


def test_per_layer_fetch_clear_releases_only_that_layers_budget():
    session = CaptureSession()
    session.attach(torch.nn.Identity(), dict.fromkeys(COMPONENTS, ()))
    session._available_layers["hidden_states"] = {0, 1}
    session.enable_stream("hidden_states", budget_rows=3)
    store = session._streams["hidden_states"]
    append_rows(store, layer=0)
    preserved = append_rows(store, layer=1, values=torch.full((3, 2), 9.0))
    assert set(session.fetch_stream("hidden_states", layers=[0])) == {0}
    assert 0 not in store.layer_names
    replacement = append_rows(
        store, layer=0, request="b", values=torch.full((3, 2), 7.0), name="new.0"
    )
    append_rows(store, layer=1, request="b", values=torch.zeros(3, 2))
    raw = session.fetch_stream("hidden_states", clear=False)
    tensors, labels = deserialize_captured(raw)
    torch.testing.assert_close(tensors[0], replacement)
    torch.testing.assert_close(tensors[1], preserved)
    assert labels[0].req_ids == ["b"] * 3
    assert labels[1].req_ids == ["a"] * 3
    assert raw[0]["layer_name"] == "new.0"
    assert session.stream_status("hidden_states")["tokens_stored"] == 3


def test_per_request_drain_releases_budget_and_preserves_other_rows():
    store = StreamStore(StreamConfig(budget_rows=4))
    append_rows(store, request="a", values=torch.ones(2, 2))
    append_rows(store, request="b", values=torch.full((2, 2), 2.0))
    drained, labels = deserialize_captured(
        store.serialize(req_ids=["a"], clear_selected=True)
    )
    assert labels[0].req_ids == ["a", "a"]
    torch.testing.assert_close(drained[0], torch.ones(2, 2))
    append_rows(store, request="c", values=torch.full((2, 2), 3.0))
    tensors, labels = deserialize_captured(store.serialize())
    assert labels[0].req_ids == ["b", "b", "c", "c"]
    assert tensors[0][:, 0].tolist() == [2, 2, 3, 3]
    assert store.tokens_stored == 4


def test_paged_fetch_preserves_offsets_and_releases_partial_chunk_storage():
    store = StreamStore(StreamConfig(staging_bytes=64))
    values = torch.arange(22.0).reshape(11, 2)
    append_rows(store, values=values)
    original_bytes = store.storage_bytes
    preview, labels = deserialize_captured(store.serialize(max_rows=4, row_offset=2))
    torch.testing.assert_close(preview[0], values[2:6])
    assert labels[0].positions.tolist() == [2, 3, 4, 5]
    assert store.storage_bytes == original_bytes

    pages, positions = [], []
    while raw := store.serialize(max_rows=2, clear_selected=True):
        tensors, labels = deserialize_captured(raw)
        pages.append(tensors[0])
        positions.extend(labels[0].positions.tolist())
        for chunk in store.chunks.get(0, {}).values():
            # Partial drains must not retain the original page's backing store.
            assert chunk.tensor.untyped_storage().nbytes() == (
                chunk.tensor.numel() * chunk.tensor.element_size()
            )
            assert not chunk.tensor.is_pinned() and not chunk.meta.is_pinned()
    torch.testing.assert_close(torch.cat(pages), values)
    assert positions == list(range(11))
    assert store.storage_bytes == 0 and store.req_table == {}


def test_paging_filters_requests_before_applying_each_layers_row_limit():
    session = CaptureSession()
    session.attach(torch.nn.Identity(), dict.fromkeys(COMPONENTS, ()))
    session._available_layers["hidden_states"] = {0, 1, 2}
    session.enable_stream("hidden_states", staging_bytes=64)
    store = session._streams["hidden_states"]
    for layer in (0, 1):
        append_rows(store, layer=layer, request="a", values=torch.ones(3, 2))
        append_rows(store, layer=layer, request="b", values=torch.full((2, 2), 2.0))
        append_rows(store, layer=layer, request="a", values=torch.full((3, 2), 3.0))
    raw = session.fetch_stream("hidden_states", req_ids=["a"], max_rows=4)
    tensors, labels = deserialize_captured(raw)
    for layer in (0, 1):
        assert labels[layer].req_ids == ["a"] * 4
        assert tensors[layer][:, 0].tolist() == [1, 1, 1, 3]
    status = session.stream_status("hidden_states")
    assert status["layer_rows"] == {0: 4, 1: 4, 2: 0}
    assert status["pending_bytes"] == status["staging_allocation_bytes"] == 0
    tensors, labels = deserialize_captured(session.fetch_stream("hidden_states"))
    assert tensors[0][:, 0].tolist() == [2, 2, 3, 3]
    assert labels[0].req_ids == ["b", "b", "a", "a"]


@pytest.mark.parametrize(
    "options",
    [
        {"max_rows": 0},
        {"max_rows": 1.5},
        {"row_offset": -1},
        {"row_offset": True},
        {"row_offset": 1, "clear_selected": True},
    ],
)
def test_invalid_pages_fail_before_draining(options):
    store = StreamStore(StreamConfig())
    values = append_rows(store)
    with pytest.raises(ValueError, match="max_rows|row_offset"):
        store.serialize(**options)
    tensors, _ = deserialize_captured(store.serialize())
    torch.testing.assert_close(tensors[0], values)


def test_device_budget_rejects_rows_before_materialization():
    store = StreamStore(StreamConfig(device_budget_bytes=59))
    assert store.limit_rows(0, 3, row_bytes=20, device_bytes=60) == 0
    assert store.pending_bytes == store.storage_bytes == 0
    with pytest.raises(RuntimeError, match="device staging budget.*exceeded"):
        store.serialize()
    store.clear()
    assert store.limit_rows(0, 2, row_bytes=20, device_bytes=40) == 2
    store.device_reserved_bytes = 20
    assert store.limit_rows(0, 2, row_bytes=20, device_bytes=40) == 0
    with pytest.raises(RuntimeError, match="device staging budget.*exceeded"):
        store.serialize()


@pytest.mark.parametrize(
    "options",
    [
        {"staging_bytes": 0},
        {"staging_bytes": True},
        {"device_budget_bytes": -1},
        {"device_budget_bytes": 1.5},
        {"budget_rows": 1.5},
    ],
)
def test_invalid_memory_controls_fail_when_enabling_capture(options):
    with pytest.raises(
        ValueError, match="staging_bytes|device_budget_bytes|budget_rows"
    ):
        StreamConfig(**options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA D2H transfers")
def test_cuda_flush_reuses_bounded_pinned_page_and_keeps_corpus_pageable():
    store = StreamStore(StreamConfig(staging_bytes=256))
    values = torch.arange(160.0, device="cuda").reshape(40, 4)
    append_rows(store, values=values)
    staging = store._staging.data_ptr()
    assert store.staging_allocation_bytes == 256
    assert store.pending_device_bytes == 0
    for chunk in store.chunks[0].values():
        assert not chunk.tensor.is_pinned() and not chunk.meta.is_pinned()
        assert chunk.tensor.shape[0] * store.row_bytes(chunk.tensor) <= 256
    tensors, _ = deserialize_captured(store.serialize(clear_selected=True))
    torch.testing.assert_close(tensors[0], values.cpu())
    append_rows(store, request="next", values=values[:3])
    assert store._staging.data_ptr() == staging
    tensors, _ = deserialize_captured(store.serialize())
    torch.testing.assert_close(tensors[0], values[:3].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA staging")
def test_device_budget_flushes_previous_layers_before_next_selection():
    store = StreamStore(StreamConfig(device_budget_bytes=112, staging_bytes=256))
    index = store.req_index("a")
    values = torch.ones((2, 4), device="cuda")
    labels = torch.tensor(
        [[index, 0, 10], [index, 1, 11]], device="cuda", dtype=torch.int32
    )
    store.append(0, values, labels, "layer.0")
    assert store.pending_device_bytes == 56
    next_index = store.req_index("b")
    assert store.limit_rows(1, 3, row_bytes=28, device_bytes=84) == 3
    assert store.pending_device_bytes == 0
    assert store.req_table[next_index] == "b"
    next_labels = labels.clone()
    next_labels[:, 0] = next_index
    store.append(1, values, next_labels, "layer.1")
    store.flush()
    tensors, metadata = deserialize_captured(store.serialize())
    torch.testing.assert_close(tensors[0], values.cpu())
    assert metadata[1].req_ids == ["b", "b"]


@pytest.mark.parametrize("shape,budget", [((2, 2), 64), ((2, 2, 2), 80)])
def test_byte_budget_counts_layers_dtype_and_labels_and_rejects_incomplete_capture(
    shape, budget
):
    store = StreamStore(StreamConfig(dtype="float16", budget_bytes=budget))
    index = store.req_index("a")
    values = torch.ones(shape, dtype=torch.float32)
    meta = torch.tensor([[index, 0, 10], [index, 1, 11]], dtype=torch.int32)
    # Both layers include every feature dimension, FP16 values and row labels.
    for layer in (0, 1):
        store.append(layer, values, meta, f"layer.{layer}")
    assert store.storage_bytes == budget
    store.flush()
    assert store.storage_bytes == budget
    assert store.limit_rows(2, 1, store.row_bytes(values)) == 0
    with pytest.raises(RuntimeError, match="CPU storage budget.*exceeded"):
        store.serialize()
    store.clear()
    assert store.storage_bytes == 0
    append_rows(store, values=values)
    store.serialize(req_ids=["a"], clear_selected=True)
    assert store.storage_bytes == 0
    store.config.budget_bytes = budget // 4 - 1
    store.append(0, values[:1], meta[:1], "layer.0")
    assert store.storage_bytes == 0
    with pytest.raises(RuntimeError, match="CPU storage budget.*exceeded"):
        store.serialize()


@pytest.mark.parametrize(
    "meta",
    [None, torch.zeros(2, 2, dtype=torch.int32), torch.zeros(2, 3)],
)
def test_append_rejects_missing_or_malformed_labels_before_staging(meta):
    store = StreamStore(StreamConfig(budget_rows=2))
    with pytest.raises(RuntimeError, match="requires int32.*row labels"):
        store.append(0, torch.ones(2, 2), meta, "layer.0")
    store.flush()
    assert store.serialize() == {}


@pytest.mark.parametrize("invalid", ["missing", "row_count", "request_index"])
def test_deserialize_rejects_incomplete_labels_instead_of_dropping_alignment(invalid):
    store = StreamStore(StreamConfig(budget_rows=3))
    append_rows(store)
    raw = store.serialize()
    if invalid == "missing":
        raw[0]["meta"] = None
    elif invalid == "row_count":
        raw[0]["meta"]["positions"] = b""
    else:
        raw[0]["meta"]["req_table"] = []
    with pytest.raises(ValueError, match="row labels|request indices"):
        deserialize_captured(raw)


def test_request_drain_reclaims_ids_and_preserves_active_capture_history():
    store = StreamStore(StreamConfig(budget_rows=6))
    append_rows(store, request="a")
    append_rows(store, layer=1, request="b")
    append_rows(store, request="c")
    raw = store.serialize(req_ids=["a"], clear_selected=True)
    assert raw[0]["meta"]["req_table"] == ["a"]
    assert set(store.req_table.values()) == {"b", "c"}
    raw = store.serialize()
    assert raw[0]["meta"]["req_table"] == ["c"]
    assert raw[1]["meta"]["req_table"] == ["b"]
    _, labels = deserialize_captured(raw)
    assert labels[0].req_ids == ["c"] * 3
    assert labels[1].req_ids == ["b"] * 3
    store.mark_elided("a")
    assert not store.elided_reqs  # Active requests may resume after draining.
    store.finish_requests({"a"})
    assert "a" not in store._captured_requests


def test_draining_completed_requests_does_not_accumulate_historical_ids():
    store = StreamStore(StreamConfig(budget_rows=3))
    for index in range(20):
        request = f"request-{index}"
        append_rows(store, request=request)
        store.finish_requests({request})
        raw = store.serialize(req_ids=[request], clear_selected=True)
        assert raw[0]["meta"]["req_table"] == [request]
        assert store.req_table == {} and store._req_index == {}
        assert not store._captured_requests
        assert store.serialize() == {}


def test_recycling_request_slots_preserves_labels_shared_by_layer_plans():
    store = StreamStore(StreamConfig(budget_rows=3))
    append_rows(store, request="a")
    meta = torch.tensor(
        [[store.req_index("b"), 0, 10], [store.req_index("c"), 0, 20]],
        dtype=torch.int32,
    )
    for layer in (1, 2):
        store.append(layer, torch.ones(2, 2), meta, f"layer.{layer}")
    store.flush()
    store.drop_layers([0])
    _, labels = deserialize_captured(store.serialize())
    assert set(store.req_table.values()) == {"b", "c"}
    assert labels[1].req_ids == labels[2].req_ids == ["b", "c"]
    append_rows(store, request="d")
    _, labels = deserialize_captured(store.serialize())
    assert labels[0].req_ids == ["d"] * 3
    assert labels[1].req_ids == labels[2].req_ids == ["b", "c"]


def test_request_drain_preserves_order_without_copying_unrelated_activations():
    store = StreamStore(StreamConfig(budget_rows=8))
    request_ids = ["a-01234567", "b", "a-01234567", "c", "b"]
    meta = torch.tensor(
        [[store.req_index(rid), pos, 10 + pos] for pos, rid in enumerate(request_ids)],
        dtype=torch.int32,
    )
    store.append(0, torch.arange(10.0).reshape(5, 2), meta, "layer.0")
    store.flush()
    preserved = {
        index: chunk.tensor.data_ptr()
        for index, chunk in store.chunks[0].items()
        if store.req_table[chunk.request_index] != "a-01234567"
    }
    tensors, labels = deserialize_captured(
        store.serialize(req_ids=["a"], clear_selected=True)
    )
    assert labels[0].req_ids == ["a-01234567"] * 2
    assert labels[0].positions.tolist() == [0, 2]
    assert tensors[0][:, 0].tolist() == [0, 4]
    assert {
        index: chunk.tensor.data_ptr() for index, chunk in store.chunks[0].items()
    } == preserved
    tensors, labels = deserialize_captured(store.serialize())
    assert labels[0].req_ids == ["b", "c", "b"]
    assert tensors[0][:, 0].tolist() == [2, 6, 8]
    assert store.tokens_stored == 3


def test_capture_failure_rejects_failed_request_but_allows_other_request_drain():
    session = CaptureSession()
    session.attach(torch.nn.Identity(), dict.fromkeys(COMPONENTS, ()))
    session._available_layers["hidden_states"] = {0}
    session.enable_stream("hidden_states", budget_rows=6)
    store = session._streams["hidden_states"]
    session.fail_requests({"bad-01234567": "overlapping steering selections"})
    # A failed request may still run a forward to finish its scheduled batch.
    append_rows(store, request="bad-01234567")
    expected = append_rows(store, request="good")
    session.finish_requests({"bad-01234567", "good"})
    with pytest.raises(RuntimeError, match="overlapping steering selections"):
        session.fetch_stream("hidden_states", req_ids=["bad"])
    tensors, labels = deserialize_captured(
        session.fetch_stream("hidden_states", req_ids=["good"])
    )
    torch.testing.assert_close(tensors[0], expected)
    assert labels[0].req_ids == ["good"] * 3
    with pytest.raises(RuntimeError, match="failed request"):
        session.fetch_stream("hidden_states")
    session.clear_stream("hidden_states")
    assert session.fetch_stream("hidden_states") == {}
