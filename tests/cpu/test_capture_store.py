# SPDX-License-Identifier: Apache-2.0
"""Capture storage preserves labels, wire dtypes and reusable row budgets."""

import pytest
import torch
from vllm.model_hooks.capture.serialization import deserialize_captured
from vllm.model_hooks.capture.session import CaptureSession
from vllm.model_hooks.capture.store import StreamConfig, StreamStore


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


@pytest.mark.parametrize("dtype", ["bool", "int16", "complex64", "missing"])
def test_unsupported_wire_dtype_is_rejected_when_enabling_stream(dtype):
    with pytest.raises(ValueError, match="unsupported capture storage dtype"):
        StreamConfig(dtype=dtype)


def test_per_layer_fetch_clear_releases_only_that_layers_budget():
    session = CaptureSession()
    session._attached = True
    session._hooked_layers["hidden_states"] = {0, 1}
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
    session._attached = True
    session._hooked_layers["hidden_states"] = {0}
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
