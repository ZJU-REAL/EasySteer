#!/usr/bin/env python3
"""Benchmark: eager per-request steering vs server-level CUDA graphs.

Starts two servers sequentially, sends N sequential requests to each,
and compares throughput.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_eager_vs_cudagraphs.py \
        --vector /path/to/vector.gguf \
        --n 20 --max-tokens 200
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

import httpx

MODEL = os.environ.get("STEER_TEST_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TARGET_LAYERS = list(range(10, 26))
PORT = 8019

PROMPTS = [
    "Describe a rainy Monday morning.",
    "Write a short story about a lost cat.",
    "Explain how a bicycle works.",
    "What makes a good cup of coffee?",
    "Describe the view from a mountaintop.",
]


def wait_for_server(port: int, timeout: float = 180.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"http://localhost:{port}/v1/models", timeout=2.0)
            if r.status_code == 200:
                print(f"  Server ready after {time.time() - t0:.1f}s")
                return
        except httpx.ConnectError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"Server not ready after {timeout}s")


def start_server(
    vector_path: str,
    scale: float,
    *,
    server_level: bool,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(PORT),
        "--gpu-memory-utilization", "0.4",
        "--max-model-len", "512",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
    ]
    if server_level:
        import json as _json
        spec = {"vectors": [{
            "source": vector_path,
            "scale": scale,
            "layers": TARGET_LAYERS,
            "normalize": True,
            "apply": {"phases": ["prompt", "generation"]},
        }]}
        cmd += ["--steering-config", _json.dumps(spec)]
    else:
        cmd += [
            "--enable-steer-vector",
            "--enforce-eager",
        ]
    mode = "server-level + CUDA graphs" if server_level else "per-request + eager"
    log = f"/tmp/vllm_bench_{'cg' if server_level else 'eager'}.log"
    print(f"  Starting {mode} server (log: {log})")
    log_file = open(log, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    proc._log = log_file  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    proc._log.close()  # type: ignore[attr-defined]


def bench(
    vector_path: str,
    scale: float,
    n: int,
    max_tokens: int,
    *,
    per_request: bool,
) -> dict:
    url = f"http://localhost:{PORT}"
    client = httpx.Client(timeout=300.0)
    total_tokens = 0

    # Warmup
    for i in range(2):
        body: dict = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPTS[0]}],
            "max_tokens": 10,
            "temperature": 0.8,
        }
        if per_request:
            body["steering"] = {"vectors": [{
                "source": vector_path,
                "scale": scale,
                "layers": TARGET_LAYERS,
                "normalize": True,
                "apply": {"phases": ["prompt", "generation"]},
            }]}
        r = client.post(f"{url}/v1/chat/completions", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"Warmup failed: {r.status_code} {r.text[:300]}")

    # Benchmark
    t0 = time.perf_counter()
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }
        if per_request:
            body["steering"] = {"vectors": [{
                "source": vector_path,
                "scale": scale,
                "layers": TARGET_LAYERS,
                "normalize": True,
                "apply": {"phases": ["prompt", "generation"]},
            }]}
        r = client.post(f"{url}/v1/chat/completions", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"Request {i} failed: {r.status_code} {r.text[:300]}")
        total_tokens += r.json()["usage"]["completion_tokens"]

    elapsed = time.perf_counter() - t0
    client.close()

    return {
        "n_requests": n,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 1),
        "avg_latency_ms": round(elapsed / n * 1000, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", required=True)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    vector_path = str(os.path.abspath(args.vector))
    assert os.path.exists(vector_path), f"Vector not found: {vector_path}"

    print("=" * 60)
    print("Benchmark: eager per-request vs server-level CUDA graphs")
    print("=" * 60)
    print(f"  Requests: {args.n}, max_tokens: {args.max_tokens}")
    print()

    # ── Eager (per-request) ──────────────────────────────────────
    print("[1/2] Eager mode (per-request steering)")
    proc = start_server(vector_path, args.scale, server_level=False)
    try:
        wait_for_server(PORT)
        eager = bench(
            vector_path, args.scale, args.n, args.max_tokens,
            per_request=True,
        )
    finally:
        stop_server(proc)
    print(f"  {eager}")
    print()

    # ── CUDA graphs (server-level) ───────────────────────────────
    print("[2/2] CUDA graphs (server-level steering)")
    proc = start_server(vector_path, args.scale, server_level=True)
    try:
        wait_for_server(PORT)
        cg = bench(vector_path, args.scale, args.n, args.max_tokens, per_request=False)
    finally:
        stop_server(proc)
    print(f"  {cg}")
    print()

    # ── Summary ──────────────────────────────────────────────────
    speedup = (
        eager["elapsed_s"] / cg["elapsed_s"]
        if cg["elapsed_s"] > 0
        else float("inf")
    )
    tps_speedup = (
        cg["tokens_per_sec"] / eager["tokens_per_sec"]
        if eager["tokens_per_sec"] > 0
        else float("inf")
    )

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    hdr = f"  {'':>25} {'Eager':>12} {'CUDA graphs':>12}"
    print(f"{hdr} {'Speedup':>10}")
    print(f"  {'-'*60}")
    e_tok = eager['total_tokens']
    c_tok = cg['total_tokens']
    print(f"  {'Tokens':>25} {e_tok:>12} {c_tok:>12}")
    e_t = eager['elapsed_s']
    c_t = cg['elapsed_s']
    print(
        f"  {'Wall time':>25} {e_t:>11.1f}s"
        f" {c_t:>11.1f}s {speedup:>9.2f}x"
    )
    e_tps = eager['tokens_per_sec']
    c_tps = cg['tokens_per_sec']
    print(
        f"  {'Throughput (tok/s)':>25} {e_tps:>12.1f}"
        f" {c_tps:>12.1f} {tps_speedup:>9.2f}x"
    )
    e_lat = eager['avg_latency_ms']
    c_lat = cg['avg_latency_ms']
    print(
        f"  {'Avg latency (ms)':>25}"
        f" {e_lat:>12.1f} {c_lat:>12.1f}"
    )


if __name__ == "__main__":
    main()
