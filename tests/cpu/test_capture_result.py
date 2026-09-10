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
    assert calls == ["start_capture", "fetch_captured", "stop_capture"]
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
