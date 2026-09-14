"""Capture adapters retain sample and layer order without repeated slicing."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_hooks.capture.serialization import CaptureMeta

from easysteer.hidden_states import CaptureResult


def test_nested_conversion_slices_each_sample_once():
    labels = CaptureMeta(
        req_ids=["a", "b", "a"],
        positions=torch.tensor([0, 0, 1], dtype=torch.int32),
        token_ids=torch.tensor([10, 20, 11], dtype=torch.int32),
    )
    result = CaptureResult(
        {7: torch.arange(12).reshape(3, 4), 2: torch.ones(3, 4)},
        meta={7: labels, 2: labels},
        outputs=[SimpleNamespace(request_id="a"), SimpleNamespace(request_id="b")],
    )
    with patch.object(result, "sample", wraps=result.sample) as sample:
        nested = result.to_nested()
    assert sample.call_count == 2
    assert len(nested) == 2 and all(len(layers) == 2 for layers in nested)
    torch.testing.assert_close(nested[0][0], result.layers[2][[0, 2]])
    torch.testing.assert_close(nested[1][1], result.layers[7][[1]])


@pytest.mark.parametrize("invalid", ["missing", "partial_layers", "row_count"])
def test_result_rejects_missing_or_incomplete_row_labels(invalid):
    labels = CaptureMeta(
        req_ids=["a"],
        positions=torch.tensor([0], dtype=torch.int32),
        token_ids=torch.tensor([10], dtype=torch.int32),
    )
    layers = {2: torch.ones(1, 4), 7: torch.ones(1, 4)}
    meta = {2: labels, 7: labels}
    error = ValueError
    if invalid == "missing":
        meta = None
    elif invalid == "partial_layers":
        del meta[7]
    else:
        layers[7] = torch.ones(2, 4)
        error = RuntimeError
    with pytest.raises(error, match="row labels"):
        CaptureResult(layers, meta, [SimpleNamespace(request_id="a")])


def test_result_rejects_reversed_sample_order_in_another_layer():
    """A shared sample index must never return another request's layer rows."""
    first = CaptureMeta(
        req_ids=["a", "b"],
        positions=torch.tensor([0, 0], dtype=torch.int32),
        token_ids=torch.tensor([10, 20], dtype=torch.int32),
    )
    reversed_labels = CaptureMeta(
        req_ids=["b", "a"],
        positions=first.positions.flip(0),
        token_ids=first.token_ids.flip(0),
    )
    with pytest.raises(RuntimeError, match="row labels differ between layers"):
        CaptureResult(
            {2: torch.tensor([[10.], [20.]]), 7: torch.tensor([[200.], [100.]])},
            {2: first, 7: reversed_labels},
            [SimpleNamespace(request_id="a"), SimpleNamespace(request_id="b")],
        )


@pytest.mark.parametrize("stream", ["hidden_states", "attention_heads"])
def test_capture_helper_preserves_prompts_and_forwards_steering(stream):
    """The engine plans cache reads; the helper must not re-key prompts."""
    from easysteer.hidden_states import capture

    prompts = [{"prompt_token_ids": [10, 11], "cache_salt": "caller-salt"}]
    seen = {}
    calls = []

    def generate(inputs, **kwargs):
        seen.update(inputs=inputs, **kwargs)
        return [SimpleNamespace(request_id="0")]

    def rpc(method, args, kwargs):
        calls.append(method)
        if method == "capture_status":
            return [{"tokens_dropped": 0}]
        return [{}] if method == "fetch_captured" else [True]

    llm = SimpleNamespace(
        generate=generate, llm_engine=SimpleNamespace(collective_rpc=rpc)
    )
    steering = object()
    result = capture(llm, prompts, stream=stream, steering=steering, temperature=0.25)
    assert seen["inputs"] is prompts
    assert prompts == [{"prompt_token_ids": [10, 11], "cache_salt": "caller-salt"}]
    assert seen["steering"] is steering
    assert seen["sampling_params"].temperature == 0.25
    assert calls == [
        "capture_status",
        "start_capture",
        "capture_status",
        "fetch_captured",
        "stop_capture",
    ]
    assert result.labelled and result.sample(0) == {}
    assert result.sample_positions(0) == result.sample_token_ids(0) == []


def test_capture_result_retains_head_layout_without_guessing_from_hidden_size():
    labels = CaptureMeta(
        req_ids=["sample"], positions=torch.tensor([0]), token_ids=torch.tensor([10]),
    )
    rows = torch.arange(12.).reshape(1, 12)
    layout = {"width": 12, "num_heads": 4, "head_size": 3}
    result = CaptureResult(
        {7: rows}, {7: labels}, [SimpleNamespace(request_id="sample")],
        layouts={7: layout},
    )
    assert result.layouts[7] == layout
    torch.testing.assert_close(result.sample(0)[7], rows)
    with pytest.raises(ValueError, match="layout does not match"):
        CaptureResult(
            {7: rows[:, :8]}, {7: labels}, result.outputs, layouts={7: layout}
        )


def test_capture_cache_policy_tracks_streams_and_request_overrides():
    from vllm.model_hooks.capture.policy import CaptureRequestPolicy

    policy = CaptureRequestPolicy()
    prompt = [10, 11, 12, 13]
    policy.record_rpc(
        "start_capture", ("hidden_states",), {"select": {"generation": "all"}}
    )
    assert not policy.skip_prefix_read(prompt, None)
    assert policy.skip_prefix_read(prompt, {"hidden_states": {"prompt": "all"}})
    policy.record_rpc(
        "start_capture",
        (),
        {"stream": "router_logits", "select": {"prompt_positions": [0]}},
    )
    assert policy.skip_prefix_read(prompt, None)
    assert not policy.skip_prefix_read(
        prompt, {"router_logits": {"prompt_positions": [-1]}}
    )
    policy.record_rpc("stop_capture", ("router_logits",), None)
    assert not policy.skip_prefix_read(prompt, None)
    policy.record_rpc("start_capture", ("hidden_states",), {})
    assert policy.skip_prefix_read(prompt, None)
    policy.record_rpc("stop_capture", ("hidden_states",), None)
    assert not policy.skip_prefix_read(prompt, None)


def test_capture_admission_normalizes_without_mutating_callers_selection():
    from vllm.model_hooks.capture.policy import normalize_capture_select

    wire = {"hidden_states": {"prompt_positions": ["-1"]}}
    normalized = normalize_capture_select(wire)
    assert normalized["hidden_states"]["prompt_positions"] == [-1]
    assert wire["hidden_states"]["prompt_positions"] == ["-1"]


def test_capture_rpc_uses_one_configuration_snapshot_during_await():
    import asyncio

    from vllm.model_hooks.capture.policy import CaptureRequestPolicy
    from vllm.v1.engine.async_llm import AsyncLLM

    original = {"select": {"generation": "all"}}
    policy = CaptureRequestPolicy()

    async def rpc(method, timeout, args, config):
        original["select"].clear()
        original["select"]["prompt"] = "all"
        await asyncio.sleep(0)
        assert config == {"select": {"generation": "all"}}
        return [True]

    engine = SimpleNamespace(
        input_processor=SimpleNamespace(capture_policy=policy),
        engine_core=SimpleNamespace(collective_rpc_async=rpc),
    )
    asyncio.run(
        AsyncLLM.collective_rpc(
            engine, "start_capture", args=("hidden_states",), kwargs=original
        )
    )
    assert not policy.skip_prefix_read([10, 11, 12], None)


@pytest.mark.parametrize("async_engine", [False, True])
def test_failed_capture_rpc_does_not_change_admission_policy(async_engine):
    """A rejected worker activation must not disable cache reads afterwards."""
    import asyncio

    from vllm.model_hooks.capture.policy import CaptureRequestPolicy
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine

    policy = CaptureRequestPolicy()

    def fail(*args):
        raise RuntimeError("capture unavailable")

    async def async_fail(*args):
        fail(*args)

    engine = SimpleNamespace(
        input_processor=SimpleNamespace(capture_policy=policy),
        engine_core=SimpleNamespace(
            collective_rpc=fail, collective_rpc_async=async_fail
        ),
    )
    with pytest.raises(RuntimeError, match="capture unavailable"):
        if async_engine:
            asyncio.run(
                AsyncLLM.collective_rpc(
                    engine, "start_capture", args=("hidden_states",)
                )
            )
        else:
            LLMEngine.collective_rpc(engine, "start_capture", args=("hidden_states",))
    assert not policy.skip_prefix_read([10, 11], None)


def test_selected_token_reads_preserve_sample_order_without_copying_all_rows():
    from easysteer.steer import extract_token_hiddens

    labels = CaptureMeta(
        ["b", "a", "a"], torch.tensor([8, 9, 2]), torch.tensor([18, 19, 12])
    )
    tensor = torch.arange(12.0).reshape(3, 4)
    result = CaptureResult(
        {7: tensor},
        {7: labels},
        [SimpleNamespace(request_id="a"), SimpleNamespace(request_id="b")],
    )
    assert result.token(0, 7, -1).data_ptr() == tensor[1].data_ptr()
    assert result.sample_rows(1, 7).data_ptr() == tensor[0].data_ptr()
    with patch.object(result, "to_nested", side_effect=AssertionError("full copy")):
        pos, neg = extract_token_hiddens(result, [0], [1], token_pos=-1)
    torch.testing.assert_close(torch.from_numpy(pos[7][0]), tensor[1])
    torch.testing.assert_close(torch.from_numpy(neg[7][0]), tensor[0])
    assert result.sample_positions(0) == [2, 9]


def test_capture_batches_slices_prompt_configuration_together_and_yields_lazily():
    import easysteer.hidden_states.capture_result as module

    prompts = [f"prompt-{i}" for i in range(5)]
    selections = [{"prompt_positions": [i]} for i in range(5)]
    steering = [object() for _ in prompts]
    calls = []

    def capture(llm, batch, **kwargs):
        calls.append((batch, kwargs))
        return batch

    with patch.object(module, "capture", capture):
        batches = module.capture_batches(
            object(),
            prompts,
            batch_size=2,
            per_prompt_selects=selections,
            steering=steering,
        )
        assert not calls
        assert list(batches) == [prompts[:2], prompts[2:4], prompts[4:]]
    for start, (batch, kwargs) in zip(range(0, 5, 2), calls):
        end = start + len(batch)
        assert kwargs["per_prompt_selects"] == selections[start:end]
        assert kwargs["steering"] == steering[start:end]


@pytest.mark.parametrize(
    "failure", ["start_capture", "generate", "dropped", "fetch_captured"]
)
def test_capture_failures_stop_the_stream_without_returning_partial_rows(failure):
    from easysteer.hidden_states import capture

    calls = []

    def rpc(method, args, kwargs):
        calls.append(method)
        if method == failure:
            raise RuntimeError("capture failed")
        if method == "capture_status":
            return [{"tokens_dropped": int(failure == "dropped")}]
        return [True]

    def generate(*args, **kwargs):
        if failure == "generate":
            raise RuntimeError("generation failed")
        return []

    llm = SimpleNamespace(
        generate=generate, llm_engine=SimpleNamespace(collective_rpc=rpc)
    )
    with pytest.raises(RuntimeError):
        capture(llm, ["prompt"])
    assert calls[-1] == "stop_capture"
    if failure == "dropped":
        assert "fetch_captured" not in calls


def _topology(rank, **overrides):
    return {
        "tp_rank": rank, "tp_size": 2,
        "pp_size": 1, "dp_size": 1, "pcp_size": 1, "dcp_size": 1,
        "sequence_parallel": False, "sequence_parallel_moe": False,
        "expert_parallel": False, **overrides,
    }


def _capture_shard(rank, *, kind="feature_shard", dtype=torch.float32):
    from vllm.model_hooks.capture.serialization import serialize_capture_layer

    tensor = torch.arange(rank * 4, rank * 4 + 4, dtype=dtype).reshape(2, 2)
    labels = torch.tensor([[0, 0, 10], [0, 1, 11]], dtype=torch.int32)
    wire = serialize_capture_layer(tensor, labels, {0: "sample"}, "model.layers.7")
    wire["shard"] = {
        "kind": kind, "tp_rank": rank, "tp_size": 2,
        "feature_start": rank * 2 if kind == "feature_shard" else 0,
        "global_width": 4 if kind == "feature_shard" else 2,
    }
    if kind == "feature_shard":
        wire["layout"] = {"width": 2, "num_heads": 2, "head_size": 1}
    return {7: wire}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_capture_assembles_global_attention_heads_independent_of_reply_order(dtype):
    from vllm.capture import assemble_captured

    tensors, meta, layouts = assemble_captured(
        [_capture_shard(1, dtype=dtype), _capture_shard(0, dtype=dtype)], tp_size=2
    )
    torch.testing.assert_close(
        tensors[7], torch.tensor([[0, 1, 4, 5], [2, 3, 6, 7]], dtype=dtype)
    )
    assert layouts[7] == {"width": 4, "num_heads": 4, "head_size": 1}
    assert meta[7].req_ids == ["sample", "sample"]
    assert meta[7].positions.tolist() == [0, 1]
    assert meta[7].token_ids.tolist() == [10, 11]


def test_capture_exports_only_one_replicated_owner_without_scaling():
    from vllm.capture import assemble_captured

    tensors, meta, layouts = assemble_captured(
        [{}, _capture_shard(0, kind="replicated")], tp_size=2
    )
    torch.testing.assert_close(tensors[7], torch.arange(4.).reshape(2, 2))
    assert len(meta[7]) == 2 and not layouts
    assert assemble_captured([{}, {}], tp_size=2) == ({}, {}, {})


def test_capture_assembly_compares_request_identity_instead_of_local_table_indices():
    from vllm.capture import assemble_captured

    results = [_capture_shard(0), _capture_shard(1)]
    labels = results[1][7]["meta"]
    labels["req_table"] = ["unused-local-request", "sample"]
    labels["req_idx"] = torch.ones(2, dtype=torch.int32).numpy().tobytes()
    tensors, metadata, _ = assemble_captured(results, tp_size=2)
    assert tuple(tensors[7].shape) == (2, 4)
    assert metadata[7].req_ids == ["sample", "sample"]


def test_capture_assembly_accepts_original_single_worker_wire_format():
    from vllm.capture import assemble_captured

    raw = _capture_shard(0)
    del raw[7]["shard"]
    tensors, _, layouts = assemble_captured([raw], tp_size=1)
    assert tuple(tensors[7].shape) == (2, 2)
    assert layouts[7] == raw[7]["layout"]


@pytest.mark.parametrize(
    "failure",
    [
        "missing_worker", "missing_shard", "duplicate_rank", "missing_metadata",
        "wrong_tp_size", "overlap", "gap", "global_width", "dtype",
        "head_size", "request", "position", "token",
    ],
)
def test_capture_assembly_rejects_incomplete_or_misaligned_attention_shards(failure):
    from vllm.capture import assemble_captured

    results = [_capture_shard(0), _capture_shard(1)]
    info = results[1][7]
    if failure == "missing_worker":
        results.pop()
    elif failure == "missing_shard":
        results[1] = {}
    elif failure == "duplicate_rank":
        info["shard"]["tp_rank"] = 0
    elif failure == "missing_metadata":
        del info["shard"]
    elif failure == "wrong_tp_size":
        info["shard"]["tp_size"] = 3
    elif failure in ("overlap", "gap"):
        info["shard"]["feature_start"] = 1 if failure == "overlap" else 3
    elif failure == "global_width":
        info["shard"]["global_width"] = 6
    elif failure == "dtype":
        results[1] = _capture_shard(1, dtype=torch.bfloat16)
    elif failure == "head_size":
        info["layout"].update(head_size=2, num_heads=1)
    elif failure == "request":
        info["meta"]["req_table"] = ["different-request"]
    else:
        key = "positions" if failure == "position" else "token_ids"
        info["meta"][key] = torch.tensor([1, 2], dtype=torch.int32).numpy().tobytes()
    with pytest.raises(ValueError):
        assemble_captured(results, tp_size=2)


@pytest.mark.parametrize("owner_ranks", [(0, 0), (0, 1), (1,)])
def test_capture_assembly_rejects_duplicate_replicas_or_missing_owner(owner_ranks):
    from vllm.capture import assemble_captured

    results = [_capture_shard(rank, kind="replicated") for rank in owner_ranks]
    results.extend({} for _ in range(2 - len(results)))
    with pytest.raises(ValueError):
        assemble_captured(results, tp_size=2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_rank", 0), ("tp_size", 3), ("pp_size", 2), ("dp_size", 2),
        ("pcp_size", 2), ("dcp_size", 2), ("sequence_parallel", True),
        ("sequence_parallel_moe", True), ("expert_parallel", True),
    ],
)
def test_capture_rejects_unsupported_topology_before_starting_workers(field, value):
    from easysteer.hidden_states import capture

    calls = []

    def rpc(method, **kwargs):
        calls.append(method)
        return [
            {"topology": _topology(0)},
            {"topology": _topology(1, **{field: value})},
        ]

    llm = SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=rpc))
    with pytest.raises(ValueError):
        capture(llm, ["prompt"])
    assert calls == ["capture_status"]


@pytest.mark.parametrize("per_prompt", [False, True])
def test_capture_validates_selectors_before_contacting_workers(per_prompt):
    from easysteer.hidden_states import capture

    def rpc(*args, **kwargs):
        pytest.fail("Invalid selectors must not reach workers")

    kwargs = (
        {"per_prompt_selects": [{"prompt": "invalid"}]}
        if per_prompt else {"select": {"prompt": "invalid"}}
    )
    llm = SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=rpc))
    with pytest.raises(ValueError):
        capture(llm, ["prompt"], **kwargs)


@pytest.mark.parametrize("failure", [None, "dropped", "fetch", "generate"])
def test_capture_helper_assembles_workers_and_observes_nonzero_rank_failures(failure):
    from easysteer.hidden_states import capture

    calls = []

    def rpc(method, **kwargs):
        calls.append(method)
        if method == "capture_status":
            return [
                {"topology": _topology(rank), "tokens_dropped": int(
                    rank == 1 and failure == "dropped" and "start_capture" in calls
                )}
                for rank in (1, 0)
            ]
        if method == "fetch_captured":
            if failure == "fetch":
                raise RuntimeError("rank 1 capture failed")
            return [_capture_shard(1), _capture_shard(0)]
        return [True, True]

    def generate(*args, **kwargs):
        if failure == "generate":
            raise RuntimeError("generation failed")
        return [SimpleNamespace(request_id="sample")]

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(collective_rpc=rpc), generate=generate
    )
    if failure is None:
        result = capture(llm, ["prompt"], stream="attention_heads")
        assert result.layouts[7]["width"] == 4
        torch.testing.assert_close(
            result.sample_rows(0, 7), torch.tensor([[0., 1., 4., 5.], [2., 3., 6., 7.]])
        )
    else:
        with pytest.raises(RuntimeError):
            capture(llm, ["prompt"], stream="attention_heads")
    assert calls[-1] == "stop_capture"
    if failure == "dropped":
        assert "fetch_captured" not in calls


def test_capture_preserves_original_error_if_cleanup_also_fails():
    from easysteer.hidden_states import capture

    def rpc(method, **kwargs):
        if method == "capture_status":
            return [{"tokens_dropped": 0}]
        if method == "stop_capture":
            raise RuntimeError("cleanup failed")
        raise ValueError("original start failed")

    llm = SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=rpc))
    with pytest.raises(ValueError, match="original start failed") as exc:
        capture(llm, ["prompt"])
    assert str(exc.value.__cause__) == "cleanup failed"
