# SPDX-License-Identifier: Apache-2.0
"""Frontend defaults resolve to ordinary requests without worker state or RPCs."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from vllm.exceptions import VLLMClientError
from vllm.model_hooks.steering.api import (
    ApplySpec,
    SteeringSpec,
    VectorSpec,
    to_engine_request,
)
from vllm.model_hooks.steering.defaults import DefaultSteeringState
from vllm.model_hooks.steering.payloads import DirectionVector
from vllm.model_hooks.steering.request import config_fingerprint
from vllm.v1.engine.input_processor import InputProcessor


def make_spec(scale=1.0, algorithm="direct"):
    return SteeringSpec(
        vectors=[
            VectorSpec(
                data=DirectionVector({1: np.ones(4, dtype=np.float32)}),
                algorithm=algorithm,
                scale=scale,
                layers=[1],
                apply=ApplySpec(prompt="all", generation="all"),
            )
        ]
    )


@pytest.fixture
def processor():
    """Use real default/admission logic without constructing model or tokenizer."""
    proc = object.__new__(InputProcessor)
    proc.vllm_config = SimpleNamespace(
        steer_vector_config=SimpleNamespace(
            algorithms=["direct"],
            multi_vector=False,
            require_preload=False,
            graph_mode="in_graph",
            graph_max_rank=32,
        ),
        parallel_config=SimpleNamespace(_api_process_count=1),
        kv_transfer_config=None,
    )
    proc._steer_preloaded_paths = set()
    proc._steer_preloaded_payloads = set()
    proc._default_steering = DefaultSteeringState()
    proc.model_config = SimpleNamespace(get_hidden_size=lambda: 4)
    proc._steering_model_info = {"hidden_states": {1: 4}, "router_logits": {1: 4}}
    return proc


def test_default_override_off_and_clear_share_request_path(processor):
    processor.set_default_steering(make_spec(scale=1.0))
    inherited = processor.resolve_steering_request(None)
    override = processor.resolve_steering_request(to_engine_request(make_spec(scale=2)))
    assert inherited.vectors[0].scale == 1.0
    assert override.vectors[0].scale == 2.0
    assert processor.resolve_steering_request(False) is None
    assert config_fingerprint(inherited) == config_fingerprint(
        to_engine_request(make_spec(scale=1.0))
    )
    processor.set_default_steering(None)
    assert processor.get_default_steering() == {"active": False}
    assert processor.resolve_steering_request(None) is None
    assert inherited.vectors[0].scale == 1.0


def test_default_snapshot_isolated_from_author_and_admitted_requests(processor):
    spec = make_spec()
    processor.set_default_steering(spec)
    admitted = processor.resolve_steering_request(None)
    fingerprint = config_fingerprint(admitted)
    spec.vectors[0].scale = 8
    spec.vectors[0].data.layers[1][:] = 999
    admitted.vectors[0].apply_spec["prompt"] = None
    admitted.vectors[0].payload["extra"]["unrelated"] = True
    fresh = processor.resolve_steering_request(None)
    assert config_fingerprint(fresh) == fingerprint
    assert fresh.vectors[0].apply_spec["prompt"] == "all"
    assert "unrelated" not in fresh.vectors[0].payload["extra"]
    # Large immutable buffers can be shared; mutable routing metadata cannot.
    key = next(iter(fresh.vectors[0].payload["tensors"]))
    assert (
        fresh.vectors[0].payload["tensors"][key]["data"]
        is (admitted.vectors[0].payload["tensors"][key]["data"])
    )
    processor.set_default_steering(make_spec(scale=2))
    assert fresh.vectors[0].scale == 1
    assert processor.resolve_steering_request(None).vectors[0].scale == 2


def test_explicit_request_is_detached_before_admission(processor):
    original = to_engine_request(make_spec())
    admitted = processor.resolve_steering_request(original)
    original.vectors[0].target_layers.append(7)
    original.vectors[0].apply_spec["generation"] = None
    assert admitted.vectors[0].target_layers == [1]
    assert admitted.vectors[0].apply_spec["generation"] == "all"


def test_rejected_replacement_keeps_previous_snapshot(processor):
    processor.set_default_steering(make_spec())
    old = config_fingerprint(processor.resolve_steering_request(None))
    with pytest.raises(VLLMClientError, match="erase.*not.*declared"):
        processor.set_default_steering(make_spec(algorithm="erase"))
    assert config_fingerprint(processor.resolve_steering_request(None)) == old
    with pytest.raises(TypeError, match="SteeringSpec or None"):
        processor.set_default_steering(False)
    assert config_fingerprint(processor.resolve_steering_request(None)) == old


def test_default_update_requires_one_frontend(processor):
    processor.vllm_config.parallel_config._api_process_count = 2
    with pytest.raises(VLLMClientError, match="api-server-count=1"):
        processor.set_default_steering(make_spec())
    assert processor.get_default_steering() == {"active": False}


def test_default_status_omits_binary_memory_weights(processor):
    processor.set_default_steering(make_spec())
    status = processor.get_default_steering()
    assert status["active"] and status["data_omitted"]
    assert "data" not in status["spec"]["vectors"][0]
    assert status["payloads"][0]["kind"] == "direction"
    assert status["payloads"][0]["sha256"] == (
        processor.resolve_steering_request(None).vectors[0].payload_sha256
    )
    json.dumps(status)
    status["spec"]["vectors"][0]["scale"] = 999
    assert processor.get_default_steering()["spec"]["vectors"][0]["scale"] == 1


def test_source_default_snapshots_file_and_preload_checks_content(processor, tmp_path):
    path = tmp_path / "router.json"
    path.write_text(
        json.dumps({"layer_configs": {"1": {"expert_ids": [1], "mode": "activate"}}})
    )
    spec = SteeringSpec(
        vectors=[
            VectorSpec(
                source=str(path),
                algorithm="moe_router",
                apply=ApplySpec(generation="all"),
            )
        ]
    )
    cfg = processor.vllm_config.steer_vector_config
    cfg.algorithms = ["moe_router"]
    cfg.require_preload = True
    first = to_engine_request(spec)
    with pytest.raises(VLLMClientError, match="not preloaded"):
        processor.set_default_steering(spec)
    processor.note_steer_vectors_preloaded(
        [first.vectors[0].payload], first.vectors[0].algorithm, [str(path)]
    )
    processor.set_default_steering(spec)
    admitted = processor.resolve_steering_request(None)
    fingerprint = config_fingerprint(admitted)
    path.write_text(
        json.dumps({"layer_configs": {"1": {"expert_ids": [2], "mode": "deactivate"}}})
    )
    with pytest.raises(VLLMClientError, match="current content"):
        processor.set_default_steering(spec)
    assert config_fingerprint(processor.resolve_steering_request(None)) == fingerprint
    assert admitted.vectors[0].payload["extra"]["layers"]["1"]["expert_ids"] == [1]
    status = processor.get_default_steering()
    assert status["spec"]["vectors"][0]["source"] == str(path)
    assert "data_omitted" not in status


def test_python_batch_preserves_none_and_false():
    from vllm.entrypoints.llm import _resolve_steering

    resolved = _resolve_steering([make_spec(), None, False])
    assert resolved[0].vectors[0].scale == 1.0
    assert resolved[1] is None and resolved[2] is False
    with pytest.raises(TypeError, match="None, or False"):
        _resolve_steering(True)
    with pytest.raises(TypeError, match="None, or False"):
        _resolve_steering([True])


@pytest.mark.parametrize("chat", [False, True])
def test_http_requests_allow_override_inherit_off_and_reject_true(chat, processor):
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    request_cls = ChatCompletionRequest if chat else CompletionRequest
    kwargs = (
        {"messages": [{"role": "user", "content": "hello"}]}
        if chat
        else {"prompt": "hello"}
    )
    kwargs["model"] = "test"
    serving = SimpleNamespace(engine_client=SimpleNamespace(input_processor=processor))
    for choice in (None, False):
        request = request_cls(**kwargs, steering=choice)
        assert GenerateBaseServing._maybe_get_steer_vector(serving, request) is False
    request = request_cls(**kwargs, steering=make_spec())
    assert (
        GenerateBaseServing._maybe_get_steer_vector(serving, request).vectors[0].scale
        == 1
    )
    with pytest.raises(ValidationError):
        request_cls(**kwargs, steering=True)


def test_sync_and_async_default_updates_use_same_frontend_validation(processor):
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine

    engine = SimpleNamespace(input_processor=processor)
    LLMEngine.set_default_steering(engine, make_spec())
    assert LLMEngine.get_default_steering(engine)["active"]
    asyncio.run(AsyncLLM.set_default_steering(engine, make_spec(scale=2)))
    assert AsyncLLM.get_default_steering(engine)["spec"]["vectors"][0]["scale"] == 2
    with pytest.raises(TypeError, match="SteeringSpec or None"):
        asyncio.run(AsyncLLM.set_default_steering(engine, False))
    asyncio.run(AsyncLLM.set_default_steering(engine, None))
    assert LLMEngine.get_default_steering(engine) == {"active": False}


def test_http_default_endpoint_sets_clears_and_preserves_rejected_state(processor):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vllm.entrypoints.serve.steering.api_router import attach_router

    processor.vllm_config.steer_vector_config.algorithms = ["moe_router"]

    async def set_default(spec):
        processor.set_default_steering(spec)

    app = FastAPI()
    app.state.engine_client = SimpleNamespace(
        set_default_steering=set_default,
        get_default_steering=processor.get_default_steering,
    )
    attach_router(app)
    client = TestClient(app)
    assert client.get("/v1/steering").json() == {"active": False}
    spec = {
        "vectors": [
            {
                "algorithm": "moe_router",
                "layers": [1],
                "params": {"expert_ids": [1]},
                "apply": {"generation": "all"},
            }
        ]
    }
    response = client.post("/v1/steering", json={"spec": spec})
    assert response.status_code == 200 and response.json()["active"]
    previous = response.json()
    for invalid in ({"spec": {"vectors": []}}, {"spec": False}, [], {}):
        assert client.post("/v1/steering", json=invalid).status_code == 400
        assert client.get("/v1/steering").json() == previous
    cleared = client.post("/v1/steering", json={"spec": None})
    assert cleared.status_code == 200 and cleared.json() == {"active": False}


def test_router_parameter_preload_matches_exact_request_content(processor, tmp_path):
    from vllm.v1.engine.llm_engine import LLMEngine

    path = tmp_path / "router-params.json"
    path.write_text(
        json.dumps({"layer_configs": {"1": {"expert_ids": [1], "mode": "activate"}}})
    )
    params = {"mode": "soft", "lambda": 0.5}
    spec = SteeringSpec(
        vectors=[
            VectorSpec(
                source=str(path),
                algorithm="moe_router",
                params=params,
                apply=ApplySpec(generation="all"),
            )
        ]
    )
    processor.vllm_config.steer_vector_config.algorithms = ["moe_router"]
    processor.vllm_config.steer_vector_config.require_preload = True
    preloaded = []

    def rpc(method, args):
        assert method == "preload_steer_vectors"
        (payloads,) = args
        preloaded.extend(payloads)

    engine = SimpleNamespace(input_processor=processor, collective_rpc=rpc)
    LLMEngine.preload_steer_vectors(engine, [str(path)], "moe_router")
    with pytest.raises(VLLMClientError, match="not preloaded"):
        processor.set_default_steering(spec)
    LLMEngine.preload_steer_vectors(engine, [str(path)], "moe_router", params)
    processor.set_default_steering(spec)
    request = processor.resolve_steering_request(None)
    assert request.vectors[0].payload_sha256 == preloaded[-1]["sha256"]
    assert request.vectors[0].payload_sha256 != preloaded[0]["sha256"]


@pytest.mark.parametrize("initial_default", [False, True])
def test_async_submission_freezes_before_preprocessing_await(
    processor, initial_default
):
    from vllm import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    if initial_default:
        processor.set_default_steering(make_spec(scale=1))

    class StopBeforeEngineCore(Exception):
        pass

    async def run():
        waiting = asyncio.Event()
        resume = asyncio.Event()
        seen = []

        async def supported_tasks():
            waiting.set()
            await resume.wait()
            return ("generate",)

        async def process_inputs(*args, **kwargs):
            seen.append(
                processor.resolve_steering_request(kwargs["steer_vector_request"])
            )
            raise StopBeforeEngineCore

        processor.process_inputs_async = process_inputs
        engine = SimpleNamespace(
            errored=False,
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
            ),
            input_processor=processor,
            get_supported_tasks=supported_tasks,
        )
        task = asyncio.create_task(
            AsyncLLM.add_request(
                engine,
                "pending",
                "raw prompt",
                SamplingParams(max_tokens=1),
            )
        )
        await waiting.wait()
        processor.set_default_steering(make_spec(scale=2))
        resume.set()
        with pytest.raises(StopBeforeEngineCore):
            await task
        if initial_default:
            assert seen[0].vectors[0].scale == 1
        else:
            assert seen == [None]

    asyncio.run(run())


@pytest.mark.parametrize("async_engine", [False, True])
@pytest.mark.parametrize("problem", ["missing_source", "shape", "worker"])
def test_preload_distinguishes_input_and_worker_errors(
    processor, tmp_path, monkeypatch, async_engine, problem
):
    from unittest.mock import AsyncMock, Mock

    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine

    path = tmp_path / "router.json"
    if problem != "missing_source":
        path.write_text(json.dumps({"layer_configs": {"1": {"expert_ids": [1]}}}))
    if problem == "shape":
        # Width validation follows successful source loading in both frontends.
        from vllm.model_hooks.steering.payloads import DirectionVector

        source = DirectionVector({1: [1.0, 2.0, 3.0]}).to_wire()
        monkeypatch.setattr(
            "vllm.model_hooks.steering.loading.resolve_vector_payload",
            lambda *args, **kwargs: source,
        )
    rpc = (AsyncMock if async_engine else Mock)(
        side_effect=ValueError("worker could not materialize payload")
    )
    engine = SimpleNamespace(input_processor=processor, collective_rpc=rpc)
    engine._ensure_steering_model_info = AsyncMock()
    algorithm = "direct" if problem == "shape" else "moe_router"
    error = ValueError if problem == "worker" else VLLMClientError
    message = {
        "missing_source": "not found",
        "shape": "hidden size",
        "worker": "worker could not materialize",
    }[problem]
    with pytest.raises(error, match=message):
        if async_engine:
            asyncio.run(
                AsyncLLM.preload_steer_vectors(engine, [str(path)], algorithm)
            )
        else:
            LLMEngine.preload_steer_vectors(engine, [str(path)], algorithm)
    assert rpc.call_count == (1 if problem == "worker" else 0)
    assert not processor._steer_preloaded_paths


@pytest.mark.parametrize("async_engine", [False, True])
def test_disabled_steering_preload_rejects_before_source_io_or_rpc(
    processor, monkeypatch, async_engine
):
    from unittest.mock import AsyncMock, Mock

    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine

    processor.vllm_config.steer_vector_config = None
    resolve = Mock(side_effect=AssertionError("source must not be read"))
    monkeypatch.setattr(
        "vllm.model_hooks.steering.loading.resolve_vector_payload", resolve
    )
    rpc = Mock(side_effect=AssertionError("worker must not receive RPC"))
    engine = SimpleNamespace(input_processor=processor, collective_rpc=rpc)
    engine._ensure_steering_model_info = AsyncMock()
    with pytest.raises(VLLMClientError, match="SteerVector is not enabled"):
        if async_engine:
            asyncio.run(AsyncLLM.preload_steer_vectors(engine, ["unused.gguf"]))
        else:
            LLMEngine.preload_steer_vectors(engine, ["unused.gguf"])
    resolve.assert_not_called()
    rpc.assert_not_called()


def test_attention_requests_and_preloads_use_each_layers_output_width(
    processor, monkeypatch
):
    processor.vllm_config.steer_vector_config.algorithms = ["attention_add"]
    processor._steering_model_info["attention_heads"] = {1: 6, 2: 8}
    for width in (6, 4):
        payload = DirectionVector({1: np.ones(width), 2: np.ones(8)})
        request = to_engine_request(SteeringSpec(vectors=[VectorSpec(
            data=payload, algorithm="attention_add", apply=ApplySpec(generation="all"),
        )]))
        monkeypatch.setattr(
            "vllm.model_hooks.steering.loading.resolve_vector_payload",
            lambda *args, **kwargs: payload.to_wire(),
        )
        if width == 6:
            processor._validate_steer_vector(request)
            assert processor.prepare_steering_preload(
                ["heads.gguf"], "attention_add", None
            ) == [payload.to_wire()]
        else:
            for operation in (
                lambda: processor._validate_steer_vector(request),
                lambda: processor.prepare_steering_preload(
                    ["heads.gguf"], "attention_add", None
                ),
            ):
                with pytest.raises(VLLMClientError, match="component width 6 at layer 1"):
                    operation()


def _beam_result():
    from vllm.logprobs import Logprob

    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                finish_reason="length",
                logprobs=[{3: Logprob(logprob=-0.1)}],
            )
        ]
    )


@pytest.mark.parametrize("online", [False, True])
def test_beam_rejects_steering_and_keeps_explicit_off_across_steps(processor, online):
    from vllm.entrypoints.generate.beam_search.offline import BeamSearchOfflineMixin
    from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
    from vllm.sampling_params import BeamSearchParams

    tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda tokens: "decoded")
    renderer = SimpleNamespace(get_tokenizer=lambda: tokenizer)
    prompt = {"type": "token", "prompt_token_ids": [1, 2]}
    params = BeamSearchParams(beam_width=1, max_tokens=2)
    calls = []
    processor.set_default_steering(make_spec(scale=1))

    if online:

        async def generate(*args, **kwargs):
            calls.append(kwargs["steer_vector_request"])
            processor.set_default_steering(make_spec(scale=2))
            yield _beam_result()

        serving = SimpleNamespace(
            renderer=renderer,
            engine_client=SimpleNamespace(input_processor=processor, generate=generate),
        )

        async def run():
            with pytest.raises(VLLMClientError, match="not supported with beam search"):
                await anext(
                    BeamSearchOnlineMixin.beam_search(
                        serving,
                        prompt,
                        "beam",
                        params,
                    )
                )
            results = [
                output
                async for output in BeamSearchOnlineMixin.beam_search(
                    serving,
                    prompt,
                    "beam-off",
                    params,
                    steer_vector_request=False,
                )
            ]
            assert results[0].finished

        asyncio.run(run())
    else:

        class Beam(BeamSearchOfflineMixin):
            def _preprocess_cmpl(self, prompts):
                return [prompt]

            def _render_and_run_requests(self, **kwargs):
                calls.extend(kwargs["steer_vector_requests"])
                processor.set_default_steering(make_spec(scale=2))
                return [_beam_result()]

        beam = Beam()
        beam.renderer = renderer
        beam.llm_engine = SimpleNamespace(input_processor=processor)
        with pytest.raises(VLLMClientError, match="not supported with beam search"):
            beam.beam_search([prompt], params)
        assert beam.beam_search([prompt], params, steering=False)
    assert calls == [False, False]


@pytest.mark.parametrize("chat", [False, True])
def test_http_beam_rejects_effective_steering_before_generator(processor, chat):
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    request_cls = ChatCompletionRequest if chat else CompletionRequest
    kwargs = (
        {"messages": [{"role": "user", "content": "hello"}]}
        if chat
        else {"prompt": "hello"}
    )
    kwargs.update(model="test", use_beam_search=True)
    serving = SimpleNamespace(engine_client=SimpleNamespace(input_processor=processor))
    processor.set_default_steering(make_spec())
    with pytest.raises(VLLMClientError, match="not supported with beam search"):
        GenerateBaseServing._maybe_get_steer_vector(serving, request_cls(**kwargs))
    request = request_cls(**kwargs, steering=False)
    assert GenerateBaseServing._maybe_get_steer_vector(serving, request) is False
