#!/usr/bin/env python3
"""Compare eager and in-graph steering with the same HTTP request workload.

Each mode starts its own server sequentially. Both use the selected steering
scope (per-request by default), fixed completion lengths, and the same prompts.
Reported throughput includes prefill and HTTP overhead; it is not a decode-only
kernel benchmark. Per-request steering supports CUDA graphs too.

Usage from the repository root:
    CUDA_VISIBLE_DEVICES=0 python tests/bench_eager_vs_cudagraphs.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --vector vectors/happy_diffmean.gguf --n 20 --max-tokens 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROMPTS = [
    "Alice's dog has passed away. Please comfort her.",
    "Describe a rainy Monday morning.",
    "Write a short story about a lost cat.",
    "Explain how a bicycle works.",
    "Describe the view from a mountaintop.",
]


def steering_spec(args: argparse.Namespace, scale: float | None = None) -> dict:
    return {
        "vectors": [
            {
                "source": args.vector,
                "algorithm": "direct",
                "scale": args.scale if scale is None else scale,
                "layers": args.target_layers,
                "normalize": False,
                "apply": {"prompt": "all", "generation": "all"},
            }
        ]
    }


def server_command(args: argparse.Namespace, *, eager: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_model_len),
        "--max-num-seqs",
        str(max(args.concurrency)),
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--no-async-scheduling",
        "--enable-steer-vector",
        "--steer-algorithms",
        "direct",
    ]
    if eager:
        command += ["--enforce-eager"]
    else:
        command += ["--steer-graph-mode", "in_graph"]
    if args.steering_mode == "default":
        command += ["--steering-config", json.dumps(steering_spec(args))]
    return command


def wait_for_server(proc: subprocess.Popen, port: int, timeout: float) -> None:
    import httpx

    start = time.monotonic()
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.monotonic() - start < timeout:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {proc.returncode}; see log"
                )
            try:
                if client.get(f"http://127.0.0.1:{port}/v1/models").status_code == 200:
                    return
            except httpx.RequestError:
                pass
            time.sleep(1)
    raise TimeoutError(f"server not ready after {timeout}s; see log")


def stop_server(proc: subprocess.Popen) -> None:
    # The benchmark owns this process group, including its vLLM workers.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


async def bench(args: argparse.Namespace, concurrency: int, workload: str) -> dict:
    import httpx

    def body(index: int) -> dict:
        request = {
            "model": args.model,
            "messages": [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}],
            "max_tokens": args.max_tokens,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 0,
        }
        if args.steering_mode == "per-request" and workload != "unsteered":
            request["steering"] = steering_spec(
                args, 0.0 if workload == "zero" else args.scale
            )
        return request

    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
        semaphore = asyncio.Semaphore(concurrency)

        async def request(index: int) -> tuple[int, float]:
            async with semaphore:
                start = time.perf_counter()
                response = await client.post(url, json=body(index))
                latency = time.perf_counter() - start
            response.raise_for_status()
            count = response.json()["usage"]["completion_tokens"]
            if count != args.max_tokens:
                raise RuntimeError(
                    f"fixed-length request returned {count} completion tokens"
                )
            return count, latency

        # Warm the same concurrency and steering workload being measured.
        for _ in range(args.warmup):
            await asyncio.gather(*(request(i) for i in range(concurrency)))
        start = time.perf_counter()
        samples = await asyncio.gather(*(request(i) for i in range(args.n)))
        elapsed = time.perf_counter() - start
    total_tokens = sum(count for count, _ in samples)
    latencies = sorted(latency * 1000 for _, latency in samples)
    return {
        "workload": workload,
        "concurrency": concurrency,
        "n_requests": args.n,
        "completion_tokens": total_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": total_tokens / elapsed,
        "avg_latency_ms": statistics.mean(latencies),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": latencies[
            min(len(latencies) - 1, int(len(latencies) * 0.95))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("STEER_TEST_MODEL"))
    parser.add_argument("--vector", required=True)
    parser.add_argument(
        "--target-layers", type=int, nargs="+", default=list(range(10, 26))
    )
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument(
        "--steering-mode", choices=("per-request", "default"), default="per-request"
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("unsteered", "zero", "steered"),
        default=["unsteered", "zero", "steered"],
        help="Per-request workloads; default steering mode measures steered only",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument(
        "--results-dir", type=Path, help="New directory for logs and results"
    )
    args = parser.parse_args()
    if not args.model:
        parser.error("provide --model or STEER_TEST_MODEL")
    args.vector = str(Path(args.vector).expanduser().resolve())
    if not Path(args.vector).is_file():
        parser.error(f"vector file does not exist: {args.vector}")
    if args.n <= 0 or args.warmup < 1 or not 0 < args.max_tokens < args.max_model_len:
        parser.error(
            "--n and --warmup must be positive; max tokens must fit max model length"
        )
    if min(args.concurrency) < 1 or max(args.concurrency) > args.n:
        parser.error("concurrency must be between 1 and --n")
    args.concurrency = sorted(set(args.concurrency))
    if args.steering_mode == "default":
        args.workloads = ["steered"]
    if args.results_dir:
        args.results_dir.mkdir(parents=True, exist_ok=False)
    else:
        args.results_dir = Path(tempfile.mkdtemp(prefix="easysteer-benchmark-"))
    # Refuse an occupied port; readiness must belong to the server we launch.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    print(f"Logs and results: {args.results_dir}", flush=True)
    results = {}
    environment = dict(os.environ)
    environment.pop("VLLM_STEER_TRACE_DIR", None)
    for mode, eager in (("eager", True), ("in_graph", False)):
        command = server_command(args, eager=eager)
        print(f"Benchmarking {mode}, steering={args.steering_mode}", flush=True)
        with (args.results_dir / f"{mode}.log").open("x") as log:
            proc = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
            try:
                startup = time.perf_counter()
                wait_for_server(proc, args.port, args.startup_timeout)
                results[mode] = {
                    "command": command,
                    "startup_s": time.perf_counter() - startup,
                    "measurements": [
                        asyncio.run(bench(args, concurrency, workload))
                        for concurrency in args.concurrency
                        for workload in args.workloads
                    ],
                }
            finally:
                stop_server(proc)
    results["throughput_ratio_in_graph_over_eager"] = [
        {
            "concurrency": graph["concurrency"],
            "workload": graph["workload"],
            "ratio": graph["tokens_per_sec"] / eager["tokens_per_sec"],
        }
        for eager, graph in zip(
            results["eager"]["measurements"], results["in_graph"]["measurements"]
        )
    ]
    result = {
        "settings": vars(args) | {"results_dir": str(args.results_dir)},
        "results": results,
    }
    (args.results_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
