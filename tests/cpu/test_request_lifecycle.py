"""Resource cleanup requests must remain constructible after wire changes."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from vllm.v1.engine.async_llm import AsyncLLM


def test_rejected_kv_transfer_submits_cleanup_request():
    engine = SimpleNamespace(add_request_async=AsyncMock())
    client = SimpleNamespace(engine_core=engine)
    params = {"remote_request_id": "prefill-1"}
    asyncio.run(
        AsyncLLM.notify_kv_transfer_request_rejected(
            client,
            "rejected-1",
            params,
            data_parallel_rank=2,
        )
    )
    request = engine.add_request_async.call_args.args[0]
    assert request.request_id == "rejected-1"
    assert request.abort_immediately
    assert request.steer_vector_request is None
    assert request.data_parallel_rank == 2
    assert request.sampling_params.extra_args["kv_transfer_params"] == params


def test_identical_payload_requests_share_slot_until_last_release():
    import numpy as np
    import torch
    from vllm.config import SteerVectorConfig
    from vllm.model_hooks.steering.api import to_engine_request
    from vllm.model_hooks.steering.payloads import DirectionVector
    from vllm.model_hooks.steering.worker_manager import WorkerSteeringState
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    manager = WorkerSteeringState(
        torch.device("cpu"),
        SteerVectorConfig(
            max_steer_vectors=2, graph_mode="split", algorithms=["direct"]
        ),
        hidden_size=8,
    )
    # Slot/refcount/selector accounting is independent of model loading.
    manager._distribute_config = Mock()

    def request(scale=1.0):
        spec = SteeringSpec(
            vectors=[
                VectorSpec(
                    data=DirectionVector({0: np.ones(8, dtype=np.float32)}),
                    scale=scale,
                    apply=ApplySpec(prompt="all"),
                )
            ]
        )
        return to_engine_request(spec)

    first = manager.acquire_config("a", request())
    second = manager.acquire_config("b", request())
    assert first == second
    assert manager._distribute_config.call_count == 1
    assert manager.slot_for_request("a") == manager.slot_for_request("b")
    assert manager.slot_clauses()[first] == [request().vectors[0].apply_spec]
    manager.release_config("a")
    assert manager.slot_for_request("b") == first
    assert first in manager.slot_clauses()
    manager.release_config("b")
    assert manager.slot_clauses() == {}
    assert manager.slot_for_request("b") is None
    recycled = manager.acquire_config("c", request(scale=2.0))
    assert recycled == first
    assert manager._distribute_config.call_count == 2


def test_prefix_cache_reuses_admitted_identity_and_preserves_prompt_boundary():
    import numpy as np
    from vllm.model_hooks.steering.api import to_engine_request
    from vllm.sampling_params import SamplingParams
    from vllm.steer_vectors import ApplySpec, DirectionVector, SteeringSpec, VectorSpec
    from vllm.v1.core.kv_cache_utils import _gen_steer_vector_extra_hash_keys
    from vllm.v1.request import Request

    steering = to_engine_request(SteeringSpec(vectors=[VectorSpec(
        data=DirectionVector({0: np.ones(8, dtype=np.float32)}),
        apply=ApplySpec(prompt="all"),
    )]))
    request = Request(
        request_id="steered",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
        steer_vector_request=steering,
    )
    prompt_keys = _gen_steer_vector_extra_hash_keys(request, 0, 4)
    decode_keys = _gen_steer_vector_extra_hash_keys(request, 4, 6)

    # Cache blocks reuse the scheduler's admitted identity without rehashing
    # the payload for each request's first block.
    assert prompt_keys[0] is request.steer_fingerprint
    assert prompt_keys == [request.steer_fingerprint]
    assert decode_keys == [request.steer_fingerprint, ("steer_prompt_len", 4)]


@pytest.mark.parametrize(
    "conflict, vector_count, publishes",
    [(None, 0, True), ("error", 1, True), ("error", 2, False),
     ("priority", 2, True), ("sequential", 2, True)],
)
def test_runtime_conflict_requests_never_publish_unvalidated_prefix_blocks(
    conflict, vector_count, publishes,
):
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    request = SimpleNamespace(
        steer_vector_request=(
            None if conflict is None else SimpleNamespace(
                conflict_resolution=conflict, vectors=[None] * vector_count,
            )
        ),
        skip_reading_prefix_cache=False,
    )
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.enable_caching = True
    manager.coordinator = Mock()
    manager.cache_blocks(request, 16)
    assert manager.coordinator.cache_blocks.call_count == int(publishes)
    assert manager.prefix_cache_lookup_enabled(request)


@pytest.mark.parametrize("mode", ["split", "in_graph"])
def test_worker_close_releases_model_payload_and_graph_ownership(mode):
    import gc
    import weakref

    import numpy as np
    import torch
    from vllm.config import SteerVectorConfig
    from vllm.model_hooks.components.registry import HIDDEN_STATES, ComponentTarget
    from vllm.model_hooks.steering import ops
    from vllm.model_hooks.steering.api import to_engine_request
    from vllm.model_hooks.steering.worker_manager import WorkerSteeringState
    from vllm.steer_vectors import ApplySpec, DirectionVector, SteeringSpec, VectorSpec

    worker = WorkerSteeringState(
        torch.device("cpu"),
        SteerVectorConfig(
            algorithms=["direct"], graph_mode=mode, max_steer_vectors=2,
            steer_vector_dtype="float32",
        ),
        hidden_size=4,
    )
    if mode == "in_graph":
        worker.enable_graph_mode(4, torch.float32, 8)
    module = torch.nn.Linear(4, 4)
    key = "lifecycle.layers.0"
    worker.attach_steering_hooks({
        HIDDEN_STATES: (ComponentTarget(key, 0, module),),
    })
    request = to_engine_request(SteeringSpec(vectors=[VectorSpec(
        data=DirectionVector({0: np.ones(4)}),
        apply=ApplySpec(prompt="all"),
    )]))
    worker.acquire_config("live", request)
    assert worker.model_info() == {HIDDEN_STATES: {0: 4}}
    controller = worker._controller_manager.controllers[key]
    references = [
        weakref.ref(module.weight), weakref.ref(controller),
        weakref.ref(worker.payload_cache.get(request.vectors[0].payload)[0]),
    ]
    if mode == "in_graph":
        references.extend([
            weakref.ref(worker.token_rows_buf),
            weakref.ref(controller.graph_tables["additive"]["V"]),
        ])
    del controller
    worker.close()
    worker.close()
    assert not module._forward_hooks
    del module
    gc.collect()
    assert all(reference() is None for reference in references)
    assert key not in ops._CONTROLLERS
    assert worker.list_configs() == set()
    assert worker.slot_clauses() == {}
    assert worker.slot_for_request("live") is None
    assert worker.graph_batch_entries() == {}
    assert worker.token_rows_buf is None
    assert worker.graph_masks_buf is None


@pytest.mark.parametrize("mode", ["split", "in_graph"])
def test_runner_reattachment_detaches_previous_model_hooks(mode):
    import torch
    from vllm.config import SteerVectorConfig
    from vllm.model_hooks.components.registry import HIDDEN_STATES, ComponentTarget
    from vllm.model_hooks.steering import ops
    from vllm.v1.worker.steer_vector_model_runner_mixin import (
        SteerVectorModelRunnerMixin,
    )

    runner = SteerVectorModelRunnerMixin()
    runner.device = torch.device("cpu")
    runner.vllm_config = SimpleNamespace(
        steer_vector_config=SteerVectorConfig(
            algorithms=["direct"], graph_mode=mode, steer_vector_dtype="float32",
            max_steer_vectors=2,
        ),
        model_config=SimpleNamespace(get_hidden_size=lambda: 4, dtype=torch.float32),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    old, new = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    key = "reload.layers.0"
    try:
        for module in (old, new):
            runner._attach_steering_hooks({
                HIDDEN_STATES: (ComponentTarget(key, 0, module),),
            })
        assert not old._forward_hooks
        assert len(new._forward_hooks) == 1
        assert ops._CONTROLLERS[key].hook_target is new
        assert runner.steer_vector_manager.model_info() == {HIDDEN_STATES: {0: 4}}
    finally:
        runner._close_steering()
    assert not new._forward_hooks
    assert key not in ops._CONTROLLERS

