# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible serving with steering, over real HTTP.

Boots `vllm serve` as a subprocess and exercises the online steering
surface end to end:
- the workload declaration is enforced at the CLI (a steering-enabled
  server without --steer-algorithms refuses to boot, naming the flag);
- per-request steering through the `steering` field on /v1/completions
  changes the output and zero-scale steering does not;
- an undeclared algorithm in a request is rejected with a 400 naming
  the declaration;
- /v1/steering reports no engine default; /v1/steering/vectors
  preloads and lists vectors.
"""

import os
import socket
import subprocess
import sys
import time

import pytest
import requests

from helpers import DENSE_MODEL, DENSE_VECTOR

BOOT_TIMEOUT_S = 300


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve_cmd(port, *extra):
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", DENSE_MODEL,
        "--host", "127.0.0.1", "--port", str(port),
        "--enforce-eager",
        "--gpu-memory-utilization", "0.18",
        "--max-model-len", "512",
        "--enable-steer-vector",
        *extra,
    ]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen(
        _serve_cmd(port, "--steer-algorithms", "direct"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited during boot (rc={proc.returncode})"
                )
            try:
                if requests.get(f"{base}/health", timeout=2).ok:
                    break
            except requests.ConnectionError:
                time.sleep(2)
        else:
            raise TimeoutError("server did not become healthy")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def completion(base, steering=None, max_tokens=24):
    body = {
        "model": DENSE_MODEL,
        "prompt": "The most important quality of good research is",
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


def test_declaration_required_to_boot():
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
        pytest.fail("undeclared steering server did not exit")
    assert proc.returncode != 0
    assert "steer_algorithms" in out


class TestPerRequestSteering:
    def test_steered_differs_zero_scale_matches(self, server):
        plain = completion(server).json()["choices"][0]["text"]
        steered = completion(server, steering_body(20.0))
        assert steered.ok, steered.text
        zero = completion(server, steering_body(0.0))
        assert zero.ok, zero.text
        assert steered.json()["choices"][0]["text"] != plain, (
            "per-request steering over HTTP produced no effect"
        )
        assert zero.json()["choices"][0]["text"] == plain, (
            "zero-scale steering must not change the output"
        )

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
