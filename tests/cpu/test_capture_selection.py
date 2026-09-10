"""Capture plans reuse metadata without reusing another layer or step's values."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.model_hooks.capture import selection
from vllm.model_hooks.capture.graph import CaptureGraphState
from vllm.model_hooks.capture.serialization import deserialize_captured
from vllm.model_hooks.capture.session import CaptureSession
from vllm.model_hooks.capture.store import StreamConfig, StreamStore
from vllm.model_hooks.selection.batch import BatchGeometry


def make_geometry(
    offsets=(0, 3, 4),
    computed=(0, 5),
    prompt=(3, 4),
    output=(0, 2),
    req_ids=("a", "b"),
    token_ids=(10, 11, 12, 20),
):
    qsl = np.asarray(offsets, dtype=np.int32)
    return BatchGeometry(
        query_start_loc=torch.from_numpy(qsl),
        query_start_loc_cpu=qsl,
        num_computed=torch.tensor(computed),
        num_prompt=torch.tensor(prompt),
        num_output=torch.tensor(output),
        req_ids=list(req_ids),
        token_ids=torch.tensor(token_ids),
    )


@pytest.fixture
def context(monkeypatch):
    ctx = SimpleNamespace(batch_geometry=make_geometry())
    monkeypatch.setattr("vllm.forward_context.get_forward_context", lambda: ctx)
    return ctx


def test_geometry_device_views_live_for_one_step():
    geo = make_geometry()
    view = geo.device_view()
    assert geo.device_view(torch.device("cpu")) is view
    assert view.is_decode.tolist() == [False, True]
    next_geo = make_geometry(output=(1, 2))
    next_view = next_geo.device_view()
    assert next_view is not view
    assert next_view.is_decode.tolist() == [True, True]


def test_layers_share_labels_but_own_rows_and_exclude_padding(context):
    store = StreamStore(StreamConfig(budget_rows=20))
    first = torch.arange(12.0).reshape(6, 2)
    second = torch.arange(18.0).reshape(6, 3)
    with patch.object(
        selection, "_build_row_plan", wraps=selection._build_row_plan
    ) as build:
        rows1, meta1 = selection.prepare_rows(first, store, 0)
        rows2, meta2 = selection.prepare_rows(second, store, 0)
    assert build.call_count == 1
    assert meta2 is meta1
    assert meta1.tolist() == [[0, 0, 10], [0, 1, 11], [0, 2, 12], [1, 5, 20]]
    torch.testing.assert_close(rows1, first[:4])
    torch.testing.assert_close(rows2, second[:4])
    snapshot = rows1.clone()
    first.add_(1000)
    torch.testing.assert_close(rows1, snapshot)


def test_new_step_reorders_requests_and_preserves_store_labels(context):
    store = StreamStore(StreamConfig(budget_rows=20))
    selection.prepare_rows(torch.zeros(4, 2), store, 0)
    context.batch_geometry = make_geometry(
        offsets=(0, 1, 3),
        computed=(6, 3),
        prompt=(4, 3),
        output=(3, 1),
        req_ids=("b", "a"),
        token_ids=(21, 13, 14),
    )
    rows, meta = selection.prepare_rows(torch.ones(3, 2), store, 0)
    assert rows.shape == (3, 2)
    assert meta.tolist() == [[1, 6, 21], [0, 3, 13], [0, 4, 14]]
    assert len(store._row_plans) == 1


def test_stream_overrides_refresh_on_next_step(context):
    store = StreamStore(StreamConfig(budget_rows=20))
    overrides = {
        "a": {
            "hidden_states": {"prompt_positions": [-1]},
            "router_logits": {"prompt_positions": [0]},
        },
        "b": {
            "hidden_states": {"generation_positions": [1]},
            "router_logits": {"prompt": "all"},
        },
    }
    tensor = torch.arange(8.0).reshape(4, 2)
    hidden, hidden_meta = selection.prepare_rows(
        tensor, store, 0, "hidden_states", overrides
    )
    router, router_meta = selection.prepare_rows(
        tensor, store, 0, "router_logits", overrides
    )
    torch.testing.assert_close(hidden, tensor[[2, 3]])
    torch.testing.assert_close(router, tensor[[0]])
    assert hidden_meta[:, 2].tolist() == [12, 20]
    assert router_meta[:, 2].tolist() == [10]

    # Admission may mutate the shared selection dictionary between forwards.
    overrides["a"]["hidden_states"] = {"prompt_positions": [0]}
    overrides["b"]["hidden_states"] = {"generation_positions": [0]}
    context.batch_geometry = make_geometry()
    rows, meta = selection.prepare_rows(tensor, store, 0, "hidden_states", overrides)
    torch.testing.assert_close(rows, tensor[[0]])
    assert meta[:, 2].tolist() == [10]


def test_store_and_effective_length_have_separate_plans(context):
    first = StreamStore(StreamConfig(budget_rows=20))
    second = StreamStore(StreamConfig(budget_rows=20))
    second.req_index("b")
    tensor = torch.arange(8.0).reshape(4, 2)
    _, full_meta = selection.prepare_rows(tensor, first, 0)
    short, short_meta = selection.prepare_rows(tensor[:2], first, 0)
    _, other_meta = selection.prepare_rows(tensor, second, 0)
    assert short.shape == (2, 2)
    assert short_meta[:, 2].tolist() == [10, 11]
    assert full_meta[:, 0].tolist() == [0, 0, 0, 1]
    assert other_meta[:, 0].tolist() == [1, 1, 1, 0]


@pytest.mark.parametrize("reduce", ["last", "mean"])
def test_chunked_reductions_reuse_boundaries_for_each_layers_values(context, reduce):
    context.batch_geometry = make_geometry(
        offsets=(0, 2, 3),
        token_ids=(10, 11, 20),
    )
    store = StreamStore(StreamConfig(reduce=reduce))
    tensor = torch.tensor([[1.0, 3.0], [3.0, 5.0], [7.0, 9.0]])
    with patch.object(
        selection, "_build_row_plan", wraps=selection._build_row_plan
    ) as build:
        rows, meta = selection.prepare_rows(tensor, store, 0)
        next_rows, next_meta = selection.prepare_rows(tensor + 10, store, 0)
    assert build.call_count == 1 and next_meta is meta
    if reduce == "last":
        torch.testing.assert_close(rows, tensor[[2]])
        assert meta.tolist() == [[1, 5, 20]]
    else:
        torch.testing.assert_close(rows, torch.tensor([[2.0, 4.0], [7.0, 9.0]]))
        assert meta.tolist() == [[0, -1, -1], [1, -1, -1]]
    torch.testing.assert_close(next_rows, rows + 10)


def test_mean_keeps_empty_sample_zero_row(context):
    context.batch_geometry = make_geometry(offsets=(0, 0, 2), token_ids=(20, 21))
    store = StreamStore(StreamConfig(reduce="mean"))
    tensor = torch.tensor([[2.0, 4.0], [4.0, 6.0]])
    rows, meta = selection.prepare_rows(tensor, store, 0)
    torch.testing.assert_close(rows, torch.tensor([[0.0, 0.0], [3.0, 5.0]]))
    assert meta.tolist() == [[0, -1, -1], [1, -1, -1]]


def test_last_empty_segments_never_claim_another_requests_row(context):
    context.batch_geometry = make_geometry(
        offsets=(0, 0, 2, 2),
        computed=(5, 6, 7),
        prompt=(3, 3, 3),
        output=(3, 4, 5),
        req_ids=("empty_head", "filled", "empty_tail"),
        token_ids=(20, 21),
    )
    store = StreamStore(StreamConfig(reduce="last"))
    tensor = torch.tensor([[2.0, 4.0], [4.0, 6.0]])
    rows, meta = selection.prepare_rows(tensor, store, 0)
    torch.testing.assert_close(rows, tensor[[1]])
    assert meta.tolist() == [[1, 7, 21]]


def test_last_nonfinal_prefill_does_not_register_capture_rows(context):
    context.batch_geometry = make_geometry(
        offsets=(0, 2),
        computed=(0,),
        prompt=(4,),
        output=(0,),
        req_ids=("prefill",),
        token_ids=(10, 11),
    )
    store = StreamStore(StreamConfig(reduce="last"))
    assert selection.prepare_rows(torch.ones(2, 2), store, 0) == (None, None)
    assert store.req_table == {}


def test_unmatched_selector_is_cached_only_for_its_step(context):
    store = StreamStore(StreamConfig(select={"prompt_tokens": [99]}))
    tensor = torch.ones(4, 2)
    with patch.object(
        selection, "_build_row_plan", wraps=selection._build_row_plan
    ) as build:
        assert selection.prepare_rows(tensor, store, 0) == (None, None)
        assert selection.prepare_rows(tensor, store, 0) == (None, None)
    assert build.call_count == 1
    assert store.req_table == {}
    context.batch_geometry = make_geometry(token_ids=(10, 99, 12, 20))
    rows, meta = selection.prepare_rows(tensor, store, 0)
    assert rows.shape == (1, 2)
    assert meta.tolist() == [[0, 1, 99]]


@pytest.mark.parametrize(
    "select, skipped, missing",
    [
        ({"prompt": "all"}, 0, False),
        ({"prompt": "all"}, 2, True),
        ({"generation": "all"}, 2, False),
        ({"prompt_positions": [-1]}, 3, False),
        ({"prompt_positions": [99]}, 3, False),
        ({"prompt_positions": [-2]}, 2, False),
        ({"prompt_positions": [-2]}, 3, True),
        ({"prompt_window": [-2, None]}, 2, False),
        ({"prompt_window": [-2, None]}, 3, True),
        ({"prompt_tokens": [12]}, 2, False),
        ({"prompt_tokens": [11]}, 2, True),
        ({"prompt": "all", "exclude_prompt_window": [0, 2]}, 2, False),
        ({"prompt_positions": [0], "exclude_prompt_tokens": [10]}, 2, False),
        ({"prompt_tokens": [10, 11], "exclude_prompt_positions": [0]}, 2, True),
    ],
)
def test_cache_elision_checks_actual_skipped_rows(select, skipped, missing):
    assert (
        selection.selects_skipped_prompt_rows(select, [10, 11, 12, 13], skipped)
        is missing
    )


@pytest.mark.parametrize(
    "global_select, override, missing",
    [
        ({"generation": "all"}, {"prompt": "all"}, True),
        ({"prompt": "all"}, {"generation": "all"}, False),
        ({"prompt": "all"}, {"prompt_positions": [-1]}, False),
    ],
)
def test_cache_elision_uses_request_override_for_its_stream(
    global_select, override, missing
):
    session = CaptureSession()
    session._attached = True
    session._hooked_layers["router_logits"] = {0}
    session.enable_stream("hidden_states", select=global_select)
    session.enable_stream("router_logits", select={"generation": "all"})
    session.add_request("a", {"hidden_states": override})
    session.mark_cache_elided("a", [10, 11, 12, 13], 3)
    if missing:
        with pytest.raises(RuntimeError, match="skip_reading_prefix_cache=True"):
            session.fetch_stream("hidden_states", clear=False)
        assert session.fetch_stream("hidden_states", req_ids=["other"]) == {}
    else:
        assert session.fetch_stream("hidden_states", clear=False) == {}
    assert session.fetch_stream("router_logits") == {}
    # Another request inherits the global selection, not a's override.
    session.mark_cache_elided("b", [10, 11, 12, 13], 3)
    if global_select.get("prompt") == "all":
        with pytest.raises(RuntimeError, match="incomplete"):
            session.fetch_stream("hidden_states", req_ids=["b"])
    else:
        assert session.fetch_stream("hidden_states", req_ids=["b"]) == {}


@pytest.mark.parametrize(
    "reduce, skipped, missing",
    [("all", 2, True), ("mean", 2, True), ("last", 3, False), ("last", 4, True)],
)
def test_reduction_cache_elision_preserves_already_captured_requests(
    reduce, skipped, missing
):
    session = CaptureSession()
    session._attached = True
    session.enable_stream("hidden_states", reduce=reduce, budget_rows=20)
    store = session._streams["hidden_states"]
    # A preempted request retains its previously captured rows in this store.
    store.req_index("resumed")
    session.mark_cache_elided("resumed", [10, 11, 12, 13], skipped)
    assert session.fetch_stream("hidden_states", clear=False) == {}
    session.mark_cache_elided("new", [10, 11, 12, 13], skipped)
    assert ("new" in store.elided_reqs) is missing


@pytest.mark.parametrize(
    "config, override, missing",
    [
        ({"select": {"prompt": "all"}}, None, True),
        ({"select": {"generation": "all"}}, None, False),
        ({"select": {"prompt_positions": [-1]}}, None, False),
        ({"select": {"prompt_tokens": [11]}}, None, True),
        ({"select": {"prompt": "all"}}, {"generation": "all"}, False),
        ({"select": {"generation": "all"}}, {"prompt": "all"}, True),
        ({"reduce": "last"}, {"prompt": "all"}, False),
        ({"reduce": "mean"}, {"generation": "all"}, True),
    ],
)
def test_late_capture_checks_cache_hits_and_preserves_request_selection(
    config, override, missing
):
    session = CaptureSession()
    session._attached = True
    session._hooked_layers["hidden_states"] = {0}
    session.add_request(
        "active", {} if override is None else {"hidden_states": override}
    )
    session.mark_cache_elided("active", [10, 11, 12, 13], 3)
    session.enable_stream("hidden_states", budget_rows=20, **config)
    session.prepare_batch(
        make_geometry(
            offsets=(0, 1),
            computed=(3,),
            prompt=(4,),
            output=(0,),
            req_ids=("active",),
            token_ids=(13,),
        )
    )
    if missing:
        with pytest.raises(RuntimeError, match="incomplete"):
            session.fetch_stream("hidden_states", req_ids=["active"])
        assert session.fetch_stream("hidden_states", req_ids=["other"]) == {}
    else:
        assert session.fetch_stream("hidden_states", req_ids=["active"]) == {}
    session.finish_requests({"active"})
    session.enable_stream("hidden_states", budget_rows=20)
    assert session.fetch_stream("hidden_states") == {}
    assert not session._request_selects and not session._cache_hits


def test_late_capture_ignores_completed_requests_awaiting_worker_cleanup():
    session = CaptureSession()
    session._attached = True
    session._hooked_layers["hidden_states"] = {0}
    # The prior generation is finished, but worker cleanup is carried by the
    # next scheduler output; enable cannot infer active requests from history.
    session.mark_cache_elided("finished", [10, 11, 12, 13], 3)
    session.enable_stream("hidden_states", budget_rows=20)
    assert session.fetch_stream("hidden_states", clear=False) == {}
    session.prepare_batch(
        make_geometry(
            offsets=(0, 2),
            computed=(0,),
            prompt=(2,),
            output=(0,),
            req_ids=("new",),
            token_ids=(20, 21),
        )
    )
    assert session.fetch_stream("hidden_states", clear=False) == {}
    assert not session._streams["hidden_states"].elided_reqs


def test_late_capture_rejects_only_scheduled_prompt_embedding_requests():
    session = CaptureSession()
    session._attached = True
    session._hooked_layers["hidden_states"] = {0}
    session.add_request("embed-01234567", {}, capture_supported=False)
    assert not session.any_enabled()
    session.enable_stream("hidden_states", budget_rows=20)
    # Starting capture cannot infer active requests from worker history.
    assert session.fetch_stream("hidden_states", clear=False) == {}
    session.prepare_batch(make_geometry())
    assert session.fetch_stream("hidden_states", clear=False) == {}

    geometry = make_geometry(
        offsets=(0, 1), computed=(3,), prompt=(4,), output=(0,),
        req_ids=("embed-01234567",), token_ids=(0,),
    )
    assert not session.needs_capture_for_batch(
        geometry.req_ids, [3], [1], [4], [True]
    )
    # Even an ordinary graph dispatch must record the capture failure.
    session.prepare_batch(geometry)
    with pytest.raises(RuntimeError, match="prompt embeddings"):
        session.fetch_stream("hidden_states", req_ids=["embed"])
    assert session.fetch_stream("hidden_states", req_ids=["a"]) == {}
    session.finish_requests({"embed-01234567"})
    assert not session._unsupported_requests
    session.enable_stream("hidden_states", budget_rows=20)
    assert session.fetch_stream("hidden_states") == {}


@pytest.mark.parametrize(
    "select, before_tokenization, after_tokenization",
    [
        (None, True, True),
        ({"generation_positions": [0]}, False, False),
        ({"prompt_positions": [-1], "generation": "all"}, False, False),
        ({"prompt_positions": [99]}, True, False),
        ({"prompt_tokens": [13]}, True, False),
        ({"prompt_tokens": [12]}, True, True),
        ({"prompt": "all", "exclude_prompt_window": [0, 3]}, True, False),
    ],
)
def test_recompute_policy_refines_selection_after_tokenization(
    select, before_tokenization, after_tokenization
):
    assert selection.needs_prompt_recompute(select) is before_tokenization
    assert (
        selection.needs_prompt_recompute(select, prompt_token_ids=[10, 11, 12, 13])
        is after_tokenization
    )


def test_recompute_policy_keeps_last_and_single_token_prompt_cache_safe():
    assert not selection.needs_prompt_recompute(None, reduce="last")
    assert selection.needs_prompt_recompute(None, reduce="mean")
    assert not selection.needs_prompt_recompute(None, prompt_token_ids=[10])


@pytest.mark.parametrize(
    "select, computed, scheduled, prefill, required",
    [
        ({"prompt": "all"}, 4, 1, False, False),
        ({"generation": "all"}, 0, 2, True, False),
        ({"prompt_positions": [-1]}, 0, 2, True, False),
        ({"prompt_positions": [-1]}, 2, 2, True, True),
        ({"generation_positions": [0]}, 4, 1, False, True),
        ({"generation_positions": [0]}, 5, 1, False, False),
        ({"generation_window": [2, 4]}, 5, 1, False, False),
        ({"generation_window": [2, 4]}, 6, 1, False, True),
        ({"prompt": "all", "exclude_prompt_window": [0, 2]}, 0, 2, True, False),
        ({"prompt_tokens": [99]}, 0, 2, True, True),
        ({"prompt_tokens": [99]}, 4, 1, False, False),
        ({"prompt": "all", "exclude_prompt_tokens": [10]}, 0, 1, True, True),
    ],
)
def test_capture_dispatch_proves_empty_spans_without_reading_gpu_tokens(
    select, computed, scheduled, prefill, required
):
    assert (
        selection.may_select_step_rows(select, computed, scheduled, 4, prefill)
        is required
    )


@pytest.mark.parametrize(
    "select, computed, token",
    [
        ({"prompt_tokens": [10], "exclude_prompt_positions": [0]}, 2, 10),
        ({"prompt_tokens": [99], "prompt_positions": [-1]}, 3, 10),
        ({"generation_tokens": [10], "exclude_generation_positions": [0]}, 5, 10),
        ({"generation": "all", "exclude_generation_tokens": [10]}, 4, 11),
    ],
)
def test_dispatch_upper_bound_never_skips_rows_the_actual_collector_wants(
    context, select, computed, token
):
    prefill = computed < 4
    context.batch_geometry = make_geometry(
        offsets=(0, 1),
        computed=(computed,),
        prompt=(4,),
        output=(0 if prefill else computed - 4 + 1,),
        req_ids=("a",),
        token_ids=(token,),
    )
    rows, _ = selection.prepare_rows(
        torch.ones(1, 2), StreamStore(StreamConfig(select=select)), 0
    )
    # The scheduler can budget two rows while adaptive verification runs one.
    required = selection.may_select_step_rows(select, computed, 2, 4, prefill)
    assert rows is not None
    assert required


def test_graph_dispatch_checks_every_request_override_and_enabled_stream():
    session = CaptureSession()
    session._attached = True
    session._hooked_layers = {"hidden_states": {0}, "router_logits": {0}}
    session.enable_stream("hidden_states", select={"prompt": "all"})

    def required():
        return session.needs_capture_for_batch(
            ["a", "b"], [4, 5], [1, 1], [4, 4], [False, False]
        )

    assert not required()
    session.add_request("b", {"hidden_states": {"generation_positions": [1]}})
    assert required()
    session.finish_requests({"b"})
    assert not required()
    session.enable_stream("router_logits", select={"generation": "all"})
    assert required()
    session.disable_stream("router_logits")
    assert not required()
    # Unavailable component/layer subsets have no hooks that can produce rows.
    session.enable_stream("hidden_states", layers=[9], select={"generation": "all"})
    assert not required()


def test_reduction_dispatch_requires_final_prefill_for_last_and_every_chunk_for_mean():
    assert not selection.may_select_step_rows(None, 0, 2, 4, True, reduce="last")
    assert selection.may_select_step_rows(None, 2, 2, 4, True, reduce="last")
    assert selection.may_select_step_rows(None, 4, 1, 4, False, reduce="last")
    assert selection.may_select_step_rows(None, 0, 2, 4, True, reduce="mean")


def test_graph_buffers_reuse_capacity_and_store_owns_rows_after_replay(context):
    session = CaptureSession()
    session._attached = True
    session._hooked_layers["hidden_states"] = {0}
    session.enable_stream("hidden_states", budget_rows=20)
    state = CaptureGraphState(session.graph_signature())
    session.graph_state = state
    first = torch.arange(8.0).reshape(4, 2)
    state.record("hidden_states", 0, first, "layer.0")
    buffer, _ = state.buffers[("hidden_states", 0)]
    session.collect_graph_outputs(4)
    state.record("hidden_states", 0, torch.full((2, 2), 99.0), "layer.0")
    assert state.buffers[("hidden_states", 0)][0] is buffer
    torch.testing.assert_close(buffer[:2], torch.full((2, 2), 99.0))
    tensors, _ = deserialize_captured(
        session.fetch_stream("hidden_states", clear=False)
    )
    torch.testing.assert_close(tensors[0], first)
    assert session.stream_status("hidden_states")["graph_replays"] == 1
    session.disable_stream("hidden_states")
    assert session.graph_state is state
    assert session.stream_status("hidden_states")["graph_replays"] == 1
    assert session.stream_status("hidden_states")["graph_buffer_bytes"] == 32


@pytest.fixture
def capture_runner():
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.capture_model_runner_mixin import CaptureModelRunnerMixin

    runner = CaptureModelRunnerMixin()
    runner.vllm_config = SimpleNamespace(steer_vector_config=None)
    runner.cudagraph_manager = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL,
        dispatch=lambda *args: SimpleNamespace(cg_mode=CUDAGraphMode.FULL),
    )
    runner.parallel_config = SimpleNamespace(world_size_across_dp=1)
    runner.speculative_config = None
    runner.lora_config = None
    session = runner._capture_session()
    session._attached = True
    session._hooked_layers = {"hidden_states": {0, 1}, "router_logits": {0}}
    runner.start_capture("hidden_states", layers=[0])
    return runner


def test_graph_variant_reuses_selection_changes_and_invalidates_layers_together(
    capture_runner, monkeypatch,
):
    from vllm.v1.worker.gpu import model_hook_utils

    monkeypatch.setattr(
        model_hook_utils, "_initialize_capture_graph", lambda *args: object(),
    )
    runner = capture_runner
    session = runner.capture_session
    manager = model_hook_utils.prepare_capture_graph(runner, 1, 4, None, 0, 4)
    state = session.graph_state
    state.record("hidden_states", 0, torch.ones(4, 2), "layer.0")
    runner.start_capture(
        "hidden_states", layers=[0], dtype="float16", select={"generation": "all"}
    )
    assert session.graph_state is state
    runner.clear_captured("hidden_states")
    assert session.graph_state is state
    runner.stop_capture("hidden_states")
    assert runner.capture_status("hidden_states")["graph_ready"]
    runner.start_capture("hidden_states", layers=[0], select={"prompt": "all"})
    assert session.graph_signature() == state.signature
    assert model_hook_utils.prepare_capture_graph(runner, 1, 4, None, 0, 4) is manager
    runner.start_capture("hidden_states", layers=[1], reduce="last")
    replacement = model_hook_utils.prepare_capture_graph(runner, 1, 4, None, 0, 4)
    assert replacement is not manager
    assert not state.buffers and session.graph_state is not state
    runner.start_capture("router_logits", select={"generation": "all"})
    assert (
        model_hook_utils.prepare_capture_graph(runner, 1, 4, None, 0, 4)
        is not replacement
    )


@pytest.mark.parametrize(
    "unsupported", ["parallel", "spec", "lora", "split", "nonfull"],
)
def test_capture_graph_keeps_unsupported_execution_eager(
    unsupported, capture_runner, monkeypatch,
):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu import model_hook_utils

    runner = capture_runner
    monkeypatch.setattr(
        model_hook_utils, "_initialize_capture_graph", lambda *args: object(),
    )
    assert model_hook_utils.prepare_capture_graph(runner, 1, 4, None, 0, 4)
    if unsupported == "parallel":
        runner.parallel_config.world_size_across_dp = 2
    elif unsupported == "spec":
        runner.speculative_config = object()
    elif unsupported == "lora":
        runner.lora_config = object()
    elif unsupported == "split":
        runner.vllm_config.steer_vector_config = SimpleNamespace(graph_mode="split")
    else:
        runner.cudagraph_manager.dispatch = lambda *args: SimpleNamespace(
            cg_mode=CUDAGraphMode.NONE,
        )
    assert model_hook_utils.prepare_capture_graph(runner, 1, 4, None, 0, 4) is None


def test_graph_each_forward_requires_all_outputs_despite_existing_warmup_buffers():
    state = CaptureGraphState((("hidden_states", (0, 1)),))
    state.begin_forward()
    for layer in (0, 1):
        state.record("hidden_states", layer, torch.ones(4, 2), str(layer))
    state.end_forward()
    assert len(state.buffers) == 2
    state.begin_forward()
    state.record("hidden_states", 0, torch.zeros(2, 2), "0")
    with pytest.raises(RuntimeError, match="did not write its configured outputs"):
        state.end_forward()
    state.begin_forward()
    for layer in (0, 1):
        state.record("hidden_states", layer, torch.zeros(1, 2), str(layer))
    state.end_forward()


def test_graph_rejects_duplicate_output_in_one_forward():
    state = CaptureGraphState((("hidden_states", (0,)),))
    state.begin_forward()
    state.record("hidden_states", 0, torch.ones(2, 2), "0")
    with pytest.raises(RuntimeError, match="was written twice"):
        state.record("hidden_states", 0, torch.ones(2, 2), "0")


def test_graph_initialization_failure_releases_state_and_same_signature_can_retry(
    capture_runner, monkeypatch,
):
    from vllm.v1.worker.gpu import model_hook_utils

    runner = capture_runner
    session = runner.capture_session
    attempts = []
    failure = RuntimeError("capture failed")
    model = torch.nn.Identity()

    def initialize(runner, state):
        attempts.append(state)
        with state.record_outputs(model):
            if len(attempts) == 1:
                state.buffers[("hidden_states", 0)] = (torch.ones(2, 2), "0")
                raise failure
        return object()

    monkeypatch.setattr(model_hook_utils, "_initialize_capture_graph", initialize)
    with pytest.raises(RuntimeError) as error:
        model_hook_utils.prepare_capture_graph(runner, 1, 4, None, 0, 4)
    assert error.value is failure
    assert session.graph_state is None and runner.capture_graph_manager is None
    assert not attempts[0].buffers and not attempts[0].recording
    assert not attempts[0].ready
    assert not model._forward_hooks and not model._forward_pre_hooks
    manager = model_hook_utils.prepare_capture_graph(runner, 1, 4, None, 0, 4)
    state = session.graph_state
    assert state is not attempts[0]
    assert manager is runner.capture_graph_manager and state.buffers == {}
    assert session.stream_status("hidden_states")["graph_ready"]
    assert model_hook_utils.prepare_capture_graph(runner, 1, 4, None, 0, 4) is manager
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "config, overrides",
    [
        ({}, {}),
        ({"reduce": "last"}, {}),
        ({"reduce": "mean"}, {}),
        ({"select": {"prompt_positions": [-1]}}, {}),
        ({"select": {"prompt_tokens": [11]}}, {}),
        ({"select": {"generation": "all", "exclude_generation_tokens": [20]}}, {}),
        (
            {"select": {"prompt": "all"}},
            {
                "a": {"hidden_states": {"prompt_positions": [-1]}},
                "b": {"hidden_states": {"generation_tokens": [20]}},
            },
        ),
    ],
)
def test_budget_accounting_counts_actual_selection_and_request_overrides(
    context, config, overrides
):
    store = StreamStore(StreamConfig(**config))
    count = selection.count_selected_rows(
        context.batch_geometry, store, "hidden_states", overrides
    )
    assert store.req_table == {}  # Accounting must not retain dropped requests.
    rows, _ = selection.prepare_rows(
        torch.ones(4, 2), store, 0, "hidden_states", overrides
    )
    assert count == (0 if rows is None else rows.shape[0])


def test_budget_trims_before_copy_and_full_layers_keep_ordinary_dispatch(context):
    session = CaptureSession()
    session._attached = True
    session._hooked_layers["hidden_states"] = {0}
    session.enable_stream("hidden_states", budget_rows=2)
    store = session._streams["hidden_states"]
    tensor = torch.arange(8.0).reshape(4, 2)
    session.prepare_batch(context.batch_geometry)
    rows, meta = selection.prepare_rows(tensor, store, 0)
    assert rows.shape == (2, 2)
    assert rows.untyped_storage().nbytes() == 2 * 2 * tensor.element_size()
    store.append(0, rows, meta, "layer.0")
    store.flush()
    assert store.tokens_dropped == 2
    assert not session.needs_capture_for_batch(
        ["a", "b"], [0, 5], [3, 1], [3, 4], [True, False]
    )
    session.prepare_batch(make_geometry())
    assert store.tokens_dropped == 6
    state = CaptureGraphState(session.graph_signature())
    state.record("hidden_states", 0, tensor, "layer.0")
    session.graph_state = state
    with patch("vllm.model_hooks.capture.session.prepare_rows") as prepare:
        session.collect_graph_outputs(4)
    prepare.assert_not_called()
    assert store.tokens_dropped == 6
    session.fetch_stream("hidden_states", req_ids=["a"])
    assert session.needs_capture_for_batch(
        ["a", "b"], [0, 5], [3, 1], [3, 4], [True, False]
    )
    session.collect_graph_outputs(4)
    assert store.tokens_stored == 2
    assert store.tokens_dropped == 8


def test_detach_releases_capture_graph_streams_and_request_state(capture_runner):
    from unittest.mock import Mock

    runner = capture_runner
    session = runner.capture_session
    session.add_request("a", {"hidden_states": {"prompt": "all"}})
    handle = Mock()
    session._hook_handles = [handle]
    state = CaptureGraphState(session.graph_signature())
    runner.capture_graph_manager = object()
    state.record("hidden_states", 0, torch.ones(2, 2), "layer.0")
    session.graph_state = state
    runner._detach_capture_hooks()
    runner._detach_capture_hooks()
    handle.remove.assert_called_once()
    assert not session._attached and not session.any_enabled()
    assert not session._request_selects and not session._hook_handles
    assert not any(session._hooked_layers.values())
    assert session.graph_state is None
    assert runner.capture_graph_manager is None and not state.buffers
    assert not hasattr(runner, "capture_session")


def test_capture_reattachment_releases_previous_model_and_graph(capture_runner):
    from vllm.model_hooks.components.registry import COMPONENTS, ComponentTarget

    runner = capture_runner
    previous = None
    for model in (torch.nn.Sequential(torch.nn.Identity()),
                  torch.nn.Sequential(torch.nn.Identity())):
        components = dict.fromkeys(COMPONENTS, ())
        components["hidden_states"] = (ComponentTarget("layer.0", 0, model[0]),)
        runner._attach_capture_hooks(model, components)
        assert runner.capture_graph_manager is None
        if previous is not None:
            old_model, old_session, old_state = previous
            assert not old_model._forward_hooks and not old_model[0]._forward_hooks
            assert not old_state.buffers and old_session.graph_state is None
            assert runner.capture_session is not old_session
        runner.start_capture("hidden_states", layers=[0])
        session = runner.capture_session
        state = CaptureGraphState(session.graph_signature())
        state.record("hidden_states", 0, torch.ones(2, 2), "layer.0")
        session.graph_state = state
        runner.capture_graph_manager = object()
        previous = model, session, state
    runner._detach_capture_hooks()


@pytest.mark.parametrize(
    "method", ["start_capture", "stop_capture", "fetch_captured",
               "clear_captured", "capture_status"],
)
def test_worker_capture_rpc_rejects_v1_before_calling_runner(method):
    from unittest.mock import Mock

    from vllm.v1.worker.gpu_worker import Worker

    worker = Worker.__new__(Worker)
    worker.use_v2_model_runner = False
    worker.model_runner = Mock()
    with pytest.raises(RuntimeError, match="Capture requires the V2 model runner"):
        getattr(worker, method)("hidden_states")
    assert not worker.model_runner.mock_calls


def test_worker_capture_rpc_rejects_separate_multimodal_encoder_runner():
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.mm_encoder_model_runner import MMEncoderModelRunner

    worker = Worker.__new__(Worker)
    worker.use_v2_model_runner = True
    worker.vllm_config = SimpleNamespace(is_mm_encoder_only=True)
    worker.model_runner = MMEncoderModelRunner.__new__(MMEncoderModelRunner)
    with pytest.raises(RuntimeError, match="multimodal encoder-only"):
        worker.start_capture("hidden_states")


def test_full_budget_remembers_processed_requests_until_completion(context):
    session = CaptureSession()
    session._attached = True
    session._hooked_layers["hidden_states"] = {0}
    session.enable_stream("hidden_states", budget_rows=0)
    session.prepare_batch(context.batch_geometry)
    store = session._streams["hidden_states"]
    assert store.req_table == {} and store.tokens_dropped == 4
    session.add_request("a", {})  # Resumption preserves capture history.
    session.mark_cache_elided("a", [10, 11, 12], 2)
    assert not store.elided_reqs
    session.finish_requests({"a"})
    assert store._captured_requests == {"b"}


@pytest.mark.parametrize("clear", ["fetch", "explicit"])
def test_clearing_rows_preserves_active_request_history(context, clear):
    session = CaptureSession()
    session._attached = True
    session._hooked_layers["hidden_states"] = {0}
    session.enable_stream("hidden_states", budget_rows=1)
    session.prepare_batch(context.batch_geometry)
    store = session._streams["hidden_states"]
    rows, meta = selection.prepare_rows(torch.ones(4, 2), store, 0)
    store.append(0, rows, meta, "layer.0")
    store.flush()
    if clear == "fetch":
        session.fetch_stream("hidden_states")
    else:
        session.clear_stream("hidden_states")
    assert store.tokens_stored == store.tokens_dropped == 0
    assert store.req_table == {}
    session.mark_cache_elided("a", [10, 11, 12], 2)
    assert not store.elided_reqs


def test_budgeted_owned_rows_do_not_retain_the_full_activation_storage(context):
    store = StreamStore(StreamConfig(budget_rows=1))
    values = torch.arange(12.0).reshape(6, 2)
    rows, meta = selection.prepare_rows(values, store, 0, tensor_owned=True)
    torch.testing.assert_close(rows, values[:1])
    assert rows.untyped_storage().nbytes() == rows.numel() * rows.element_size()
    assert meta.shape == (1, 3)
