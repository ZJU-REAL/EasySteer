"""Capture adapters retain sample and layer order without repeated slicing."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_hooks.capture.serialization import CaptureMeta

from easysteer.capture import CaptureResult


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
            {2: torch.tensor([[10.0], [20.0]]), 7: torch.tensor([[200.0], [100.0]])},
            {2: first, 7: reversed_labels},
            [SimpleNamespace(request_id="a"), SimpleNamespace(request_id="b")],
        )


@pytest.mark.parametrize("stream", ["hidden_states", "attention_heads"])
def test_capture_helper_preserves_prompts_and_forwards_steering(stream):
    """The engine plans cache reads; the helper must not re-key prompts."""
    from easysteer.capture import capture

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
        req_ids=["sample"],
        positions=torch.tensor([0]),
        token_ids=torch.tensor([10]),
    )
    rows = torch.arange(12.0).reshape(1, 12)
    layout = {"width": 12, "num_heads": 4, "head_size": 3}
    result = CaptureResult(
        {7: rows},
        {7: labels},
        [SimpleNamespace(request_id="sample")],
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
    from easysteer.extraction import extract_token_hiddens

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
    import easysteer.capture.api as module

    prompts = [f"prompt-{i}" for i in range(5)]
    selections = [{"prompt_positions": [i]} for i in range(5)]
    steering = [object() for _ in prompts]
    calls = []
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(collective_rpc=lambda *args, **kwargs: [{}])
    )

    def capture(llm, batch, **kwargs):
        calls.append((batch, kwargs))
        return batch

    with patch.object(module, "capture", capture):
        batches = module.capture_batches(
            llm,
            prompts,
            batch_size=2,
            per_prompt_selects=selections,
            steering=steering,
            budget_bytes=None,
        )
        assert not calls
        assert list(batches) == [prompts[:2], prompts[2:4], prompts[4:]]
    for start, (batch, kwargs) in zip(range(0, 5, 2), calls):
        end = start + len(batch)
        assert kwargs["per_prompt_selects"] == selections[start:end]
        assert kwargs["steering"] == steering[start:end]
        assert kwargs["sample_indices"] == list(range(start, end))


@pytest.mark.parametrize(
    "failure", ["start_capture", "generate", "dropped", "fetch_captured"]
)
def test_capture_failures_stop_the_stream_without_returning_partial_rows(failure):
    from easysteer.capture import capture

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
        "tp_rank": rank,
        "tp_size": 2,
        "pp_size": 1,
        "dp_size": 1,
        "pcp_size": 1,
        "dcp_size": 1,
        "sequence_parallel": False,
        "sequence_parallel_moe": False,
        "expert_parallel": False,
        **overrides,
    }


def _capture_shard(rank, *, kind="feature_shard", dtype=torch.float32):
    from vllm.model_hooks.capture.serialization import serialize_capture_layer

    tensor = torch.arange(rank * 4, rank * 4 + 4, dtype=dtype).reshape(2, 2)
    labels = torch.tensor([[0, 0, 10], [0, 1, 11]], dtype=torch.int32)
    wire = serialize_capture_layer(tensor, labels, {0: "sample"}, "model.layers.7")
    wire["shard"] = {
        "kind": kind,
        "tp_rank": rank,
        "tp_size": 2,
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
    torch.testing.assert_close(tensors[7], torch.arange(4.0).reshape(2, 2))
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
        "missing_worker",
        "missing_shard",
        "duplicate_rank",
        "missing_metadata",
        "wrong_tp_size",
        "overlap",
        "gap",
        "global_width",
        "dtype",
        "head_size",
        "request",
        "position",
        "token",
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
        ("tp_rank", 0),
        ("tp_size", 3),
        ("pp_size", 2),
        ("dp_size", 2),
        ("pcp_size", 2),
        ("dcp_size", 2),
        ("sequence_parallel", True),
        ("sequence_parallel_moe", True),
        ("expert_parallel", True),
    ],
)
def test_capture_rejects_unsupported_topology_before_starting_workers(field, value):
    from easysteer.capture import capture

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
    from easysteer.capture import capture

    def rpc(*args, **kwargs):
        pytest.fail("Invalid selectors must not reach workers")

    kwargs = (
        {"per_prompt_selects": [{"prompt": "invalid"}]}
        if per_prompt
        else {"select": {"prompt": "invalid"}}
    )
    llm = SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=rpc))
    with pytest.raises(ValueError):
        capture(llm, ["prompt"], **kwargs)


@pytest.mark.parametrize("failure", [None, "dropped", "fetch", "generate"])
def test_capture_helper_assembles_workers_and_observes_nonzero_rank_failures(failure):
    from easysteer.capture import capture

    calls = []

    def rpc(method, **kwargs):
        calls.append(method)
        if method == "capture_status":
            return [
                {
                    "topology": _topology(rank),
                    "tokens_dropped": int(
                        rank == 1 and failure == "dropped" and "start_capture" in calls
                    ),
                }
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
            result.sample_rows(0, 7),
            torch.tensor([[0.0, 1.0, 4.0, 5.0], [2.0, 3.0, 6.0, 7.0]]),
        )
    else:
        with pytest.raises(RuntimeError):
            capture(llm, ["prompt"], stream="attention_heads")
    assert calls[-1] == "stop_capture"
    if failure == "dropped":
        assert "fetch_captured" not in calls


def test_capture_preserves_original_error_if_cleanup_also_fails():
    from easysteer.capture import capture

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


class TestCaptureBatchPlanning:
    @staticmethod
    def status(width=8, rank=0, tp_size=1, kind="replicated"):
        return {
            "topology": {"tp_rank": rank, "tp_size": tp_size},
            "layouts": {7: {"width": width}},
            "shards": {
                7: {
                    "kind": kind,
                    "tp_rank": rank,
                    "tp_size": tp_size,
                    "global_width": width * tp_size
                    if kind == "feature_shard"
                    else width,
                    "feature_start": rank * width if kind == "feature_shard" else 0,
                }
            },
        }

    @staticmethod
    def estimate(prompt, select=None, **overrides):
        from easysteer.capture.planning import estimate_prompt_bytes

        options = {
            "llm": SimpleNamespace(),
            "prompt": prompt,
            "select": select,
            "stream": "hidden_states",
            "layers": [7],
            "dtype": "float16",
            "max_tokens": 1,
            "statuses": [TestCaptureBatchPlanning.status()],
        }
        options.update(overrides)
        return estimate_prompt_bytes(**options)

    def test_last_prompt_selection_does_not_reserve_the_whole_prompt(self):
        prompt = {"prompt_token_ids": [10] * 50_000}
        assert self.estimate(prompt, {"prompt_positions": [-1]}) == 8 * 2 + 12
        assert self.estimate(prompt) == 50_000 * (8 * 2 + 12)

    def test_prompt_selector_union_exclusion_and_clamped_positions(self):
        # Include 0, 2 and the clamped final index 4; token 3 excludes index 2.
        prompt = {"prompt_token_ids": [1, 2, 3, 4, 5]}
        selection = {
            "prompt_tokens": [1],
            "prompt_positions": [2, 999],
            "exclude_prompt_tokens": [3],
        }
        assert self.estimate(prompt, selection) == 2 * (8 * 2 + 12)

    def test_future_token_ids_are_conservative_but_position_exclusions_apply(self):
        # At most four generated inputs get forwarded with max_tokens=5.
        # Unknown token IDs can match three of them after excluding step 1.
        selection = {
            "generation_tokens": [123],
            "exclude_generation_tokens": [456],
            "exclude_generation_positions": [1],
        }
        assert self.estimate(
            {"prompt_token_ids": [10, 11]}, selection, max_tokens=5
        ) == 3 * (8 * 2 + 12)
        assert (
            self.estimate({"prompt_token_ids": [10, 11]}, selection, max_tokens=1) == 0
        )

    def test_selection_block_boundaries_preserve_prompt_and_decode_windows(
        self, monkeypatch
    ):
        from easysteer.capture import planning

        monkeypatch.setattr(planning, "_SELECTION_BLOCK_ROWS", 3)
        selection = {
            "prompt_window": [-5, None],
            "exclude_prompt_positions": [-2],
            "generation_window": [2, 7],
            "exclude_generation_positions": [4],
        }
        assert self.estimate(
            {"prompt_token_ids": list(range(11))}, selection, max_tokens=8
        ) == 8 * (8 * 2 + 12)

    def test_tp_attention_accounts_for_labels_and_equal_worker_budget_shares(self):
        statuses = [
            self.status(3, 0, 2, "feature_shard"),
            self.status(5, 1, 2, "feature_shard"),
        ]
        assert self.estimate(
            {"prompt_token_ids": [10, 11]}, stream="attention_heads", statuses=statuses
        ) == 2 * 2 * (5 * 2 + 12)

    def test_tp_replicated_capture_counts_only_the_owner(self):
        statuses = [self.status(8, 1, 2), self.status(8, 0, 2)]
        assert self.estimate({"prompt_token_ids": [10, 11]}, statuses=statuses) == 2 * (
            8 * 2 + 12
        )

    @pytest.mark.parametrize("config_path", ["model_config", "vllm_config"])
    def test_model_dtype_fallback_and_router_upcasting(self, config_path):
        config = SimpleNamespace(dtype=torch.bfloat16)
        engine = (
            SimpleNamespace(model_config=config)
            if config_path == "model_config"
            else SimpleNamespace(vllm_config=SimpleNamespace(model_config=config))
        )
        llm = SimpleNamespace(llm_engine=engine)
        prompt = {"prompt_token_ids": [10]}
        assert self.estimate(prompt, llm=llm, dtype=None) == 8 * 2 + 12
        assert (
            self.estimate(prompt, llm=llm, dtype=None, stream="router_logits")
            == 8 * 4 + 12
        )

    def test_text_uses_tokenizer_special_tokens(self):
        calls = []

        def encode(text, *, add_special_tokens):
            calls.append((text, add_special_tokens))
            return [1, 10, 11]

        llm = SimpleNamespace(get_tokenizer=lambda: SimpleNamespace(encode=encode))
        assert self.estimate("hello", llm=llm) == 3 * (8 * 2 + 12)
        assert calls == [("hello", True)]

    @pytest.mark.parametrize(
        "prompt",
        [
            {"prompt_token_ids": [10], "multi_modal_data": {"image": object()}},
            {"prompt_embeds": object()},
            "text without an available tokenizer",
        ],
    )
    def test_unknown_prompt_geometry_is_isolated_by_the_caller(self, prompt):
        assert self.estimate(prompt) is None

    def test_unknown_layout_or_dtype_is_not_guessed(self):
        prompt = {"prompt_token_ids": [10]}
        assert self.estimate(prompt, statuses=[{}]) is None
        assert self.estimate(prompt, dtype=None) is None


def _paged_capture_engine(dtype=torch.float32, *, failure=None, legacy=False):
    """Actual wire serialization with bounded worker pages and scrambled replies."""
    from vllm.model_hooks.capture.serialization import serialize_capture_layer

    values = torch.arange(20, dtype=dtype).reshape(5, 4)
    row_labels = torch.tensor(
        [[1, 0, 20], [0, 3, 13], [1, 2, 22], [0, 1, 11], [0, 4, 14]],
        dtype=torch.int32,
    )
    calls = []
    offset = 0
    ranks = (0,) if legacy else (1, 0)

    def rpc(method, args, kwargs):
        nonlocal offset
        calls.append((method, kwargs.copy()))
        if method == "capture_status":
            if legacy:
                return [{"tokens_dropped": 0}]
            return [
                {
                    "topology": _topology(rank),
                    "tokens_dropped": 0,
                    "layer_rows": {7: 5},
                    "layouts": {7: {"width": 2, "num_heads": 1, "head_size": 2}},
                }
                for rank in ranks
            ]
        if method == "fetch_captured":
            assert kwargs["clear"] is True
            assert legacy or kwargs["layers"] == [7]
            end = min(5, offset + kwargs.get("max_rows", 5))
            replies = []
            for rank in ranks:
                page = (
                    values[offset:end]
                    if legacy
                    else values[offset:end, rank * 2 : (rank + 1) * 2].contiguous()
                )
                labels = row_labels[offset:end].clone()
                if failure == "labels" and offset and rank == 1:
                    labels[0, 1] += 1
                if failure == "dtype" and offset:
                    page = page.to(torch.float64)
                wire = serialize_capture_layer(
                    page, labels, {0: "a", 1: "b"}, "model.layers.7"
                )
                if not legacy:
                    wire["shard"] = {
                        "kind": "feature_shard",
                        "tp_rank": rank,
                        "tp_size": 2,
                        "feature_start": rank * 2,
                        "global_width": 4,
                    }
                    wire["layout"] = {"width": 2, "num_heads": 1, "head_size": 2}
                    if failure == "layout" and offset:
                        wire["layout"] = {"width": 2, "num_heads": 2, "head_size": 1}
                replies.append({7: wire})
            offset = end
            return replies
        return [True for _ in ranks]

    def generate(prompts, **kwargs):
        return [
            SimpleNamespace(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=[10, 11] if request_id == "a" else [20, 21],
                outputs=[
                    SimpleNamespace(
                        index=0,
                        text="answer",
                        token_ids=[42],
                        finish_reason="length",
                        stop_reason=None,
                    )
                ],
            )
            for request_id, prompt in zip(("a", "b"), prompts)
        ]

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            collective_rpc=rpc, model_config=SimpleNamespace(model="test-model")
        ),
        generate=generate,
    )
    return llm, calls, values


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("on_disk", [False, True])
def test_capture_drains_tp_pages_with_labels_and_provenance(tmp_path, dtype, on_disk):
    from easysteer.capture import capture

    llm, calls, expected = _paged_capture_engine(dtype)
    directory = tmp_path / "captured" if on_disk else None
    result = capture(
        llm,
        ["first", "second"],
        stream="attention_heads",
        fetch_bytes=48,
        storage_dir=directory,
        sample_indices=[100, 101],
        select={"prompt": "all"},
    )
    torch.testing.assert_close(result.rows(7), expected)
    assert result.layouts == {7: {"width": 4, "num_heads": 2, "head_size": 2}}
    assert result.sample_indices == [100, 101]
    assert result.component == "attention_heads"
    assert result.model == "test-model"
    assert result.selection["prompt"] == "all"
    assert result.sample_positions(0) == [1, 3, 4]
    assert result.sample_token_ids(1) == [20, 22]
    torch.testing.assert_close(result.sample_rows(0, 7), expected[[3, 1, 4]])
    torch.testing.assert_close(result.sample_rows(1, 7), expected[[0, 2]])
    pages = [kwargs for method, kwargs in calls if method == "fetch_captured"]
    assert len(pages) >= 3
    assert all(1 <= kwargs["max_rows"] <= 2 for kwargs in pages)
    assert calls[-1][0] == "stop_capture"
    if on_disk:
        reloaded = CaptureResult.load(directory)
        torch.testing.assert_close(reloaded.rows(7), expected)
        assert reloaded.sample_indices == [100, 101]
        assert reloaded.outputs[0].outputs[0].text == "answer"


@pytest.mark.parametrize("failure", ["labels", "dtype", "layout"])
def test_invalid_capture_pages_fail_and_stop_every_worker(failure):
    from easysteer.capture import capture

    llm, calls, _ = _paged_capture_engine(failure=failure)
    with pytest.raises((ValueError, RuntimeError)):
        capture(llm, ["first", "second"], stream="attention_heads", fetch_bytes=48)
    assert calls[-1][0] == "stop_capture"


def test_legacy_single_worker_disk_capture_still_writes_reloadable_arrays(tmp_path):
    from easysteer.capture import capture

    llm, _, expected = _paged_capture_engine(legacy=True)
    directory = tmp_path / "legacy"
    capture(llm, ["first", "second"], storage_dir=directory)
    reloaded = CaptureResult.load(directory)
    torch.testing.assert_close(reloaded.rows(7), expected)
    assert reloaded.sample_positions(0) == [1, 3, 4]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_capture_disk_roundtrip_is_copy_on_write_and_preserves_sample_metadata(
    tmp_path, dtype
):
    import json

    import numpy as np

    labels = CaptureMeta(
        ["b", "a", "a"],
        torch.tensor([8, 9, 2], dtype=torch.int32),
        torch.tensor([18, 19, 12], dtype=torch.int32),
    )
    values = torch.tensor([[0.5, -3.25], [1.125, 4.0], [5.0, 6.0]], dtype=dtype)
    result = CaptureResult(
        {7: values},
        {7: labels},
        [SimpleNamespace(request_id="a"), SimpleNamespace(request_id="b")],
        layouts={7: {"width": 2}},
        component="hidden_states",
        model="model/name",
        selection={"prompt_positions": [-1]},
        sample_indices=[25, 26],
    )
    directory = tmp_path / "dataset"
    result.save(directory)
    with patch("numpy.load", wraps=np.load) as load:
        mapped = CaptureResult.load(directory)
    assert len(load.call_args_list) == 3
    assert all(
        call.kwargs == {"mmap_mode": "c", "allow_pickle": False}
        for call in load.call_args_list
    )
    torch.testing.assert_close(mapped.rows(7), values)
    assert mapped.rows(7).dtype == dtype
    assert mapped.sample_indices == [25, 26]
    assert mapped.selection == {"prompt_positions": [-1]}
    assert mapped.model == "model/name" and mapped.component == "hidden_states"
    assert mapped.sample_positions(0) == [2, 9]
    mapped.rows(7)[0, 0] = 99
    mapped.meta(7).positions[0] = 999
    reopened = CaptureResult.load(directory)
    torch.testing.assert_close(reopened.rows(7), values)
    assert reopened.meta(7).positions.tolist() == [8, 9, 2]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert {path.suffix for path in directory.iterdir()} == {".json", ".npy"}
    with pytest.raises(FileExistsError):
        result.save(directory)


def test_sample_chunk_iterator_bounds_noncontiguous_gathers_and_uses_views():
    values = torch.arange(18.0).reshape(9, 2)
    labels = CaptureMeta(
        ["a", "a", "b", "a", "b", "a", "a", "a", "a"],
        torch.tensor([0, 1, 0, 2, 1, 3, 4, 5, 6]),
        torch.arange(9),
    )
    result = CaptureResult(
        {7: values},
        {7: labels},
        [
            SimpleNamespace(request_id="a"),
            SimpleNamespace(request_id="b"),
            SimpleNamespace(request_id="empty"),
        ],
    )
    with patch.object(
        result, "sample_rows", side_effect=AssertionError("whole sample copy")
    ):
        chunks = list(result.iter_sample_rows(0, 7, chunk_size=2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 2, 1]
    torch.testing.assert_close(torch.cat(chunks), values[[0, 1, 3, 5, 6, 7, 8]])
    assert chunks[0].data_ptr() == values[0].data_ptr()
    assert chunks[2].data_ptr() == values[6].data_ptr()
    assert list(result.iter_sample_rows(2, 7, chunk_size=2)) == []
    with pytest.raises(ValueError, match="chunk_size"):
        list(result.iter_sample_rows(0, 7, chunk_size=0))


def test_capture_batches_byte_admission_is_lazy_and_retains_global_ids(tmp_path):
    import easysteer.capture.api as module

    consumed, calls = [], []

    def prompts():
        for index in range(6):
            consumed.append(index)
            yield {"prompt_token_ids": [index] * 3}

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            collective_rpc=lambda *args, **kwargs: [
                TestCaptureBatchPlanning.status(width=8)
            ]
        )
    )

    def fake_capture(llm, batch, **kwargs):
        calls.append((batch, kwargs))
        return kwargs["sample_indices"]

    with patch.object(module, "capture", fake_capture):
        batches = module.capture_batches(
            llm,
            prompts(),
            batch_size=32,
            layers=[7],
            dtype="float16",
            select={"prompt_positions": [-1]},
            budget_bytes=56,
            storage_dir=tmp_path,
        )
        assert consumed == calls == []
        assert next(batches) == [0, 1]
        assert consumed == [0, 1, 2]  # At most one lookahead prompt for byte admission.
        assert len(calls) == 1
        assert list(batches) == [[2, 3], [4, 5]]
    assert [kwargs["storage_dir"].name for _, kwargs in calls] == [
        "batch-000000000000",
        "batch-000000000002",
        "batch-000000000004",
    ]
    assert all(len(batch) == 2 for batch, _ in calls)


def test_capture_batches_isolates_unknown_geometry_and_rejects_oversized_prompts():
    import easysteer.capture.api as module

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            collective_rpc=lambda *args, **kwargs: [
                TestCaptureBatchPlanning.status(width=8)
            ]
        )
    )
    prompts = [
        {"prompt_token_ids": [10]},
        {"prompt_embeds": object()},
        {"prompt_token_ids": [20]},
    ]
    seen = []

    def fake_capture(llm, batch, **kwargs):
        seen.append(batch)
        return kwargs["sample_indices"]

    with patch.object(module, "capture", fake_capture):
        assert list(
            module.capture_batches(
                llm, iter(prompts), dtype="float16", budget_bytes=100
            )
        ) == [[0], [1], [2]]
        with pytest.raises(ValueError, match="Prompt 0.*exceeding"):
            list(
                module.capture_batches(
                    llm,
                    [{"prompt_token_ids": [10] * 5}],
                    dtype="float16",
                    budget_bytes=100,
                )
            )
    assert seen == [[prompts[0]], [prompts[1]], [prompts[2]]]


@pytest.mark.parametrize("field", ["per_prompt_selects", "steering"])
def test_capture_batches_rejects_mismatched_per_prompt_iterables(field):
    import easysteer.capture.api as module

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(collective_rpc=lambda *args, **kwargs: [{}])
    )
    with patch.object(module, "capture") as capture:
        with pytest.raises(ValueError, match="must match prompts"):
            list(
                module.capture_batches(
                    llm,
                    iter(["one", "two"]),
                    budget_bytes=None,
                    **{field: [{"prompt_positions": [-1]}]},
                )
            )
    capture.assert_not_called()


def test_capture_rejects_multiple_continuations_before_contacting_workers():
    from easysteer.capture import capture

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            collective_rpc=lambda *args, **kwargs: pytest.fail(
                "RPC before n validation"
            )
        )
    )
    with pytest.raises(ValueError, match="n=1|one.*continuation|single.*continuation"):
        capture(llm, ["prompt"], n=2)


def test_capture_surfaces_storage_failure_before_the_first_retained_row():
    from vllm.model_hooks.capture.store import StreamConfig, StreamStore

    from easysteer.capture import capture

    store = StreamStore(StreamConfig(budget_bytes=1))
    calls = []

    def rpc(method, args, kwargs):
        calls.append(method)
        if method == "capture_status":
            assert store.tokens_stored == 0
            return [{"tokens_dropped": 0, "layer_rows": {7: 0}}]
        if method == "fetch_captured":
            assert kwargs["max_rows"] == 1
            return [store.serialize(max_rows=kwargs["max_rows"])]
        if method == "stop_capture":
            store.clear()
        return [True]

    def generate(*args, **kwargs):
        labels = torch.tensor([[store.req_index("sample"), 0, 10]], dtype=torch.int32)
        store.append(7, torch.ones(1, 2), labels, "layer.7")
        store.flush()
        assert store.storage_bytes == 0
        return [SimpleNamespace(request_id="sample")]

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(collective_rpc=rpc), generate=generate
    )
    with pytest.raises(RuntimeError, match="CPU storage budget.*exceeded"):
        capture(llm, ["prompt"], budget_bytes=1)
    assert calls[-2:] == ["fetch_captured", "stop_capture"]
    assert store.serialize() == {}


def test_float64_capture_precision_survives_legacy_extraction_conversion():
    from easysteer.extraction import DiffMeanExtractor

    values = torch.tensor([[1.0 + 1e-9, 2.0], [1.0, 2.0]], dtype=torch.float64)
    labels = CaptureMeta(["a", "b"], torch.tensor([0, 0]), torch.tensor([10, 20]))
    captured = CaptureResult(
        {7: values},
        {7: labels},
        [SimpleNamespace(request_id="a"), SimpleNamespace(request_id="b")],
    )
    vector = DiffMeanExtractor.extract(captured, [0], [1], normalize=False)
    assert vector.directions[7][0] == pytest.approx(1e-9, rel=1e-6)
    assert vector.directions[7][1] == 0


def test_capture_preserves_global_selection_and_canonical_prompt_overrides(tmp_path):
    from easysteer.capture import capture

    llm, _, _ = _paged_capture_engine()
    overrides = [None, {"prompt_window": [0, None]}]
    directory = tmp_path / "overrides"
    with patch.object(llm, "generate", wraps=llm.generate) as generate:
        result = capture(
            llm,
            ["first", "second"],
            stream="attention_heads",
            fetch_bytes=48,
            select={"prompt": "all"},
            per_prompt_selects=overrides,
            storage_dir=directory,
        )
    forwarded = generate.call_args.kwargs["capture_select"]
    assert forwarded[0] is None
    assert forwarded[1]["attention_heads"] == result.per_prompt_selections[1]
    assert result.selection["prompt"] == "all"
    assert result.per_prompt_selections[0] is None
    assert result.per_prompt_selections[1]["prompt_window"] == [0, None]
    overrides[1]["prompt_window"][0] = 99
    assert result.per_prompt_selections[1]["prompt_window"] == [0, None]
    reloaded = CaptureResult.load(directory)
    assert reloaded.selection == result.selection
    assert reloaded.per_prompt_selections == result.per_prompt_selections


@pytest.mark.parametrize("indices", [[0], [1, 1], [-1, 2], [True, 1]])
def test_capture_rejects_invalid_sample_indices_before_contacting_workers(indices):
    from easysteer.capture import capture

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            collective_rpc=lambda *args, **kwargs: pytest.fail(
                "RPC before sample validation"
            )
        )
    )
    with pytest.raises(ValueError, match="sample_indices"):
        capture(llm, ["first", "second"], sample_indices=indices)


def test_loading_an_older_capture_without_prompt_overrides_uses_global_selection(
    tmp_path,
):
    import json

    result = CaptureResult(
        {}, {}, [SimpleNamespace(request_id="a")], selection={"prompt": "all"}
    )
    directory = tmp_path / "older-dataset"
    result.save(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["per_prompt_selections"]
    manifest_path.write_text(json.dumps(manifest))
    reloaded = CaptureResult.load(directory)
    assert reloaded.selection == {"prompt": "all"}
    assert reloaded.per_prompt_selections is None
