# SPDX-License-Identifier: Apache-2.0
"""Steering admin routes reject malformed input without hiding worker failures."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.steering.api_router import attach_router
from vllm.exceptions import VLLMValidationError


@pytest.fixture
def client_and_engine():
    app = FastAPI()
    engine = SimpleNamespace(
        preload_steer_vectors=AsyncMock(),
        list_preloaded_steer_vectors=Mock(return_value=["router.json"]),
    )
    app.state.engine_client = engine
    app.state.vllm_config = SimpleNamespace(steer_vector_config=object())
    attach_router(app)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, engine


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"paths": "vector.gguf"},
        {"paths": []},
        {"paths": [None]},
        {"paths": [123]},
        {"paths": [""]},
        {"paths": [" "]},
        {"paths": ["vector.gguf"], "algorithm": []},
        {"paths": ["vector.gguf"], "algorithm": ""},
        {"paths": ["vector.gguf"], "params": []},
    ],
)
def test_preload_rejects_malformed_body_before_engine_call(client_and_engine, body):
    client, engine = client_and_engine
    response = client.post("/v1/steering/vectors", json=body)
    assert response.status_code == 400
    assert response.json()["error"]
    engine.preload_steer_vectors.assert_not_awaited()


@pytest.mark.parametrize("path", ["/v1/steering", "/v1/steering/vectors"])
def test_invalid_json_is_a_client_error(client_and_engine, path):
    client, _ = client_and_engine
    response = client.post(
        path, content="{", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["error"]


def test_preload_preserves_algorithm_overrides(client_and_engine):
    client, engine = client_and_engine
    params = {"mode": "soft", "lambda": 0.5}
    response = client.post(
        "/v1/steering/vectors",
        json={"paths": ["router.json"], "algorithm": "moe_router", "params": params},
    )
    assert response.status_code == 200
    assert response.json() == {"preloaded": ["router.json"]}
    engine.preload_steer_vectors.assert_awaited_once_with(
        ["router.json"], "moe_router", params
    )


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (VLLMValidationError("invalid source payload"), 400),
        (RuntimeError("worker unavailable"), 500),
        (ValueError("worker materialization failed"), 500),
        (OSError("worker transport failed"), 500),
    ],
)
def test_preload_distinguishes_admission_and_worker_errors(
    client_and_engine, error, status
):
    client, engine = client_and_engine
    engine.preload_steer_vectors.side_effect = error
    response = client.post("/v1/steering/vectors", json={"paths": ["vector.gguf"]})
    assert response.status_code == status
