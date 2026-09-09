# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible serving with steering, over real HTTP.

Boots `vllm serve` as a subprocess, or reuses STEER_TEST_SERVER_URL, and
exercises the online steering surface end to end:
- the workload declaration is enforced at the CLI (a steering-enabled
  server without --steer-algorithms refuses to boot, naming the flag);
- per-request steering through the `steering` field on /v1/completions
  accepts zero-scale configs and exhibits a nonzero steering effect;
- an undeclared algorithm in a request is rejected with a 400 naming
  the declaration;
- /v1/steering reports no engine default; /v1/steering/vectors
  preloads and lists vectors.
"""

import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest
import requests

from helpers import DENSE_MODEL, DENSE_VECTOR

BOOT_TIMEOUT_S = 300
SERVER_URL = os.environ.get("STEER_TEST_SERVER_URL", "").rstrip("/")
SERVED_MODEL = os.environ.get("STEER_TEST_SERVED_MODEL", DENSE_MODEL)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve_cmd(port, *extra):
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", DENSE_MODEL,
        "--served-model-name", SERVED_MODEL,
        "--host", "127.0.0.1", "--port", str(port),
        "--enforce-eager",
        "--gpu-memory-utilization", "0.18",
        "--max-model-len", "512",
        "--max-num-batched-tokens", "512",
        "--max-num-seqs", "32",
        "--enable-steer-vector",
        *extra,
    ]


@pytest.fixture(scope="module")
def server(request, tmp_path_factory):
    if SERVER_URL:
        response = requests.get(f"{SERVER_URL}/health", timeout=10)
        response.raise_for_status()
        status = requests.get(f"{SERVER_URL}/v1/steering", timeout=10)
        status.raise_for_status()
        assert status.json() == {"active": False}, (
            "HTTP tests require a server with no default steering configured"
        )
        request.config._steer_engine_timings.append({
            "engine": "openai-server", "seconds": 0.0,
            "reused_url": SERVER_URL,
        })
        yield SERVER_URL
        return

    port = _free_port()
    prefix = os.environ.get("STEER_TEST_ARTIFACT_PREFIX")
    log_path = (
        Path(prefix + ".server.log") if prefix
        else tmp_path_factory.mktemp("openai-server") / "server.log"
    )
    log_file = log_path.open("w")
    started = time.monotonic()
    proc = subprocess.Popen(
        _serve_cmd(port, "--steer-algorithms", "direct"),
        stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy(),
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited during boot (rc={proc.returncode}); log: {log_path}"
                )
            try:
                if requests.get(f"{base}/health", timeout=2).ok:
                    break
            except requests.ConnectionError:
                time.sleep(2)
        else:
            raise TimeoutError(f"server did not become healthy; log: {log_path}")
        request.config._steer_engine_timings.append({
            "engine": "openai-server",
            "seconds": round(time.monotonic() - started, 3),
            "log": str(log_path),
        })
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_file.close()


def completion(base, steering=None, max_tokens=128):
    body = {
        "model": SERVED_MODEL,
        "prompt": (
            "<|im_start|>user\nAlice's dog has passed away. Please comfort her."
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if steering is not None:
        body["steering"] = steering
    return requests.post(f"{base}/v1/completions", json=body, timeout=120)


def steering_body(scale, algorithm="direct", source=DENSE_VECTOR):
    return {
        "vectors": [{
            "source": source,
            "algorithm": algorithm,
            "scale": scale,
            "layers": list(range(10, 26)),
            "apply": {"prompt": "all", "generation": "all"},
        }]
    }


@pytest.mark.skipif(bool(SERVER_URL), reason="reused service startup is not exercised")
def test_declaration_required_to_boot(tmp_path):
    """--enable-steer-vector without --steer-algorithms must fail fast
    at engine construction, not hang or serve."""
    port = _free_port()
    proc = subprocess.Popen(
        _serve_cmd(port),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    try:
        out, _ = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5)
        pytest.fail("undeclared steering server did not exit")
    prefix = os.environ.get("STEER_TEST_ARTIFACT_PREFIX")
    log_path = Path(prefix + ".undeclared.log") if prefix else tmp_path / "undeclared.log"
    log_path.write_text(out)
    assert proc.returncode != 0
    assert "steer_algorithms" in out


class TestPerRequestSteering:
    def test_zero_scale_is_served_and_nonzero_steering_has_effect(self, server):
        baseline = completion(server, False)
        assert baseline.ok, baseline.text
        plain = baseline.json()["choices"][0]["text"]
        steered = completion(server, steering_body(2.0))
        assert steered.ok, steered.text
        zero = completion(server, steering_body(0.0))
        assert zero.ok, zero.text
        assert steered.json()["choices"][0]["text"] != plain, (
            "per-request steering over HTTP produced no effect"
        )
        result = zero.json()
        assert len(result["choices"]) == 1
        assert result["choices"][0]["text"]
        assert result["choices"][0]["finish_reason"] in ("stop", "length")
        assert 0 < result["usage"]["completion_tokens"] <= 128

    def test_undeclared_algorithm_rejected(self, server):
        # erase accepts .gguf sources, so the spec parses fine and the
        # rejection is the declaration check, not source validation.
        resp = completion(server, steering_body(1.0, algorithm="erase"))
        assert resp.status_code == 400, resp.text
        assert "declared" in resp.text


class TestManagementEndpoints:
    def test_steering_status_no_engine_default(self, server):
        resp = requests.get(f"{server}/v1/steering", timeout=10)
        assert resp.ok
        assert resp.json() == {"active": False}

    def test_preload_and_list_vectors(self, server):
        resp = requests.post(
            f"{server}/v1/steering/vectors",
            json={"paths": [DENSE_VECTOR], "algorithm": "direct"},
            timeout=60,
        )
        assert resp.ok, resp.text
        listed = requests.get(f"{server}/v1/steering/vectors", timeout=10)
        assert listed.ok
        assert DENSE_VECTOR in listed.json()["preloaded"]

    def test_default_can_be_set_overridden_disabled_and_cleared(self, server):
        endpoint = f"{server}/v1/steering"
        update = requests.post(
            endpoint, json={"spec": steering_body(1.0)}, timeout=60,
        )
        assert update.ok, update.text
        try:
            assert requests.get(endpoint, timeout=10).json()["active"]
            for choice in (None, False, steering_body(-1.0)):
                response = completion(server, choice, max_tokens=4)
                assert response.ok, response.text
                assert len(response.json()["choices"]) == 1
        finally:
            cleared = requests.post(endpoint, json={"spec": None}, timeout=60)
            assert cleared.ok, cleared.text
        assert requests.get(endpoint, timeout=10).json() == {"active": False}
