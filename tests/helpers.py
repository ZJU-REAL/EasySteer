# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the EasySteer validation suites.

The steering trace (VLLM_STEER_TRACE_DIR) is the exact oracle used by
mechanism-level tests: it records, per engine step, the batch geometry
and the flat positions each steered layer applied to. `TraceOracle`
wraps generate() and returns the steered absolute positions.
"""

import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

# run_suites.sh checks only the models needed by the selected group.
# Direct pytest callers must set the corresponding model variable too.
DENSE_MODEL = os.path.expanduser(os.environ.get("STEER_TEST_MODEL", ""))
MOE_MODEL = os.path.expanduser(os.environ.get("STEER_TEST_MOE_MODEL", ""))
QWEN3_MODEL = os.path.expanduser(os.environ.get("STEER_TEST_QWEN3", ""))
DENSE_VECTOR = str(Path(os.environ.get(
    "STEER_TEST_VECTOR",
    Path(__file__).resolve().parents[1] / "vectors" / "happy_diffmean.gguf",
)).expanduser().resolve())


class CaptureGraphWorkerExtension:
    """Named test RPCs for an eager oracle and capture-buffer diagnostics."""

    @staticmethod
    def _graph_test_breakable_segments():
        import torch
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

        for wrapper in list(BreakableCUDAGraphWrapper._all_instances):
            for entry in wrapper.entries.values():
                if entry.capture is None:
                    continue
                segments = entry.capture.segments
                for index, segment in enumerate(segments):
                    graph = getattr(segment, "__self__", None)
                    if isinstance(graph, torch.cuda.CUDAGraph):
                        yield segments, index, graph

    def graph_test_start(self):
        """Count native replays during a test, without changing graph dispatch."""
        from unittest.mock import patch

        import torch

        assert not hasattr(self, "_graph_test_patch"), "replay probe already active"
        self._graph_test_replays = Counter()
        original = torch.cuda.CUDAGraph.replay

        def replay(graph):
            result = original(graph)
            self._graph_test_replays[id(graph)] += 1
            return result

        self._graph_test_patch = patch.object(torch.cuda.CUDAGraph, "replay", replay)
        self._graph_test_original_replay = original
        self._graph_test_replay = replay
        self._graph_test_patch.start()
        # Breakable capture caches bound methods before this probe is installed.
        for segments, index, graph in self._graph_test_breakable_segments():
            segments[index] = graph.replay
        return True

    def graph_test_read(self):
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.config.compilation import CUDAGraphMode

        assert hasattr(self, "_graph_test_patch"), "replay probe is not active"
        runner = self.model_runner
        # The ordinary manager owns FULL graphs. Capture-session graphs belong
        # to a separate manager and must not satisfy ordinary steering tests.
        full = {id(graph) for graph in runner.cudagraph_manager.graphs.values()}
        piecewise = {
            id(entry.cudagraph)
            for wrapper in list(CUDAGraphWrapper._all_instances)
            if wrapper.runtime_mode == CUDAGraphMode.PIECEWISE
            for entry in wrapper.concrete_cudagraph_entries.values()
            if entry.cudagraph is not None
        }
        piecewise.update(
            id(graph) for _, _, graph in self._graph_test_breakable_segments()
        )
        counts = self._graph_test_replays
        return {
            "rank": runner.capture_status("hidden_states")["topology"]["tp_rank"],
            "full_replays": sum(counts[key] for key in full),
            "piecewise_replays": sum(counts[key] for key in piecewise),
        }

    def graph_test_stop(self):
        if hasattr(self, "_graph_test_patch"):
            try:
                # Include segments first captured while the probe was active.
                for segments, index, graph in self._graph_test_breakable_segments():
                    replay = getattr(segments[index], "__func__", None)
                    if replay is self._graph_test_replay:
                        segments[index] = self._graph_test_original_replay.__get__(
                            graph, type(graph)
                        )
            finally:
                self._graph_test_patch.stop()
                del self._graph_test_patch
                del self._graph_test_replays
                del self._graph_test_original_replay
                del self._graph_test_replay
        return True

    def capture_test_set_eager(self, enabled):
        from vllm.config.compilation import CUDAGraphMode

        runner = self.model_runner
        if enabled:
            assert not hasattr(self, "_capture_test_idle_graph")
            manager = runner.cudagraph_manager
            assert manager is not None
            self._capture_test_idle_graph = (manager, manager.cudagraph_mode)
            # Disable capture eligibility while leaving ordinary dispatch valid
            # for steps whose selections are empty (including prompt prefill).
            manager.cudagraph_mode = CUDAGraphMode.NONE
        elif hasattr(self, "_capture_test_idle_graph"):
            manager, mode = self._capture_test_idle_graph
            manager.cudagraph_mode = mode
            del self._capture_test_idle_graph
        return True

    def capture_test_graph_state(self):
        runner = self.model_runner
        state = runner.capture_session.graph_state
        manager = runner.capture_graph_manager
        return {
            "rank": runner.capture_status("hidden_states")["topology"]["tp_rank"],
            "idle_mode": runner.cudagraph_manager.cudagraph_mode.name,
            "manager": id(manager) if manager is not None else None,
            "mode": manager.cudagraph_mode.name if manager is not None else None,
            "signature": None if state is None else [
                [stream, list(layers)] for stream, layers in state.signature
            ],
            "buffers": {} if state is None else {
                f"{stream}:{layer}": tensor.data_ptr()
                for (stream, layer), (tensor, _) in state.buffers.items()
            },
        }


@contextmanager
def graph_replay(llm, mode, minimum=1):
    """Require native ordinary-graph replay on every TP worker in this interval."""
    assert mode in ("full", "piecewise")
    rpc = llm.llm_engine.collective_rpc
    tp_size = llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size
    try:
        started = rpc("graph_test_start")
        assert len(started) == tp_size and all(started)
        yield
        results = rpc("graph_test_read")
        assert len(results) == tp_size
        assert {row["rank"] for row in results} == set(range(tp_size)), results
        counts = [row[f"{mode}_replays"] for row in results]
        assert all(count >= minimum for count in counts), results
        assert len(set(counts)) == 1, results
    finally:
        stopped = rpc("graph_test_stop")
        assert len(stopped) == tp_size and all(stopped)


_INCLUDE_KWARGS = (
    "prompt", "generation", "prompt_tokens", "prompt_positions",
    "prompt_window", "generation_tokens", "generation_positions",
    "generation_window",
)


def steering_spec(source=DENSE_VECTOR, scale=0.5, layers=(10,),
                  algorithm="direct", normalize=False, params=None,
                  conflict="priority", extra_vectors=(), **apply_kwargs):
    """Build a single-vector SteeringSpec (the common test shape).

    With no include selector among apply_kwargs the clause covers both
    phases whole, so exclude-only callers keep the old default scope.
    """
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    if not any(k in apply_kwargs for k in _INCLUDE_KWARGS):
        apply_kwargs["prompt"] = "all"
        apply_kwargs["generation"] = "all"
    vec = VectorSpec(
        source=source,
        algorithm=algorithm,
        scale=scale,
        layers=list(layers) if layers is not None else None,
        normalize=normalize,
        params=dict(params or {}),
        apply=ApplySpec(**apply_kwargs),
    )
    return SteeringSpec(vectors=[vec, *extra_vectors], conflict=conflict)


def trace_cursor(trace_dir):
    """Snapshot each worker's file independently, including replaced files."""
    return {
        path.name: (stat.st_ino, stat.st_size)
        for path in Path(trace_dir).glob("*.jsonl")
        for stat in [path.stat()]
    }


def read_trace(trace_dir, cursor, layers):
    """Read records appended since a cursor (or 0 for the entire trace).

    Step keys include the worker filename, so TP peers cannot overwrite each
    other's geometry. Apply records use the same keys for direct step lookup.
    Passing layers=None includes applies from every layer.
    """
    steps, applies = {}, []
    if cursor == 0:
        cursor = {}
    for path in sorted(Path(trace_dir).glob("*.jsonl")):
        with path.open("rb") as f:
            stat = os.fstat(f.fileno())
            inode, offset = cursor.get(path.name, (stat.st_ino, 0))
            if inode == stat.st_ino and offset <= stat.st_size:
                f.seek(offset)
            for line in f:
                rec = json.loads(line)
                rec["worker"] = path.name
                rec["step"] = (path.name, rec["step"])
                if rec["type"] == "step":
                    steps[rec["step"]] = rec
                elif rec["type"] == "apply" and (
                    layers is None or rec["layer"] in layers
                ):
                    applies.append(rec)
    return steps, applies


class TraceOracle:
    """Generate with steering and report steered absolute positions.

    Single-request batches only (asserts): positions are offset by the
    request's computed-token count at each step, so results are exact
    absolute sequence positions regardless of chunking.
    last_by_worker contains this run's validated observations, or {} when
    no trace was emitted or validation failed.
    """

    def __init__(self, llm, trace_dir):
        self.llm = llm
        self.trace_dir = trace_dir
        self.last_by_worker = {}

    def snapshot(self):
        return trace_cursor(self.trace_dir)

    def _positions_by_worker(self, steps, applies, layers):
        workers = {step["worker"] for step in steps.values()}
        expected = self.llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size
        assert len(workers) == expected, (workers, expected)
        by_worker = {
            worker: {layer: [] for layer in layers} for worker in sorted(workers)
        }
        for rec in applies:
            step = steps[rec["step"]]
            assert len(step["req_ids"]) == 1, "oracle expects single-request batches"
            num_computed = step["num_computed"][0]
            is_prefill = step["num_output"][0] == 0
            for pos in rec["positions"]:
                by_worker[rec["worker"]][rec["layer"]].append(
                    (pos + num_computed, is_prefill)
                )
        reference = next(iter(by_worker.values()))
        assert all(value == reference for value in by_worker.values()), by_worker
        self.last_by_worker = by_worker
        return reference

    def run(self, prompt_ids, sampling_params, layers=(10,), **gen_kwargs):
        """Return output and applies; explicit steering=False may emit no trace."""
        from vllm.inputs import TokensPrompt

        self.last_by_worker = {}
        start = self.snapshot()
        out = self.llm.generate(
            TokensPrompt(prompt_token_ids=list(prompt_ids)),
            sampling_params,
            use_tqdm=False,
            **gen_kwargs,
        )[0]
        steering_off = gen_kwargs.get("steering") is False
        steps, applies = read_trace(
            self.trace_dir, start, None if steering_off else layers
        )
        if steering_off:
            assert not applies, f"applies with steering=False: {applies}"
            if not steps:
                return out, {layer: [] for layer in layers}
        return out, self._positions_by_worker(steps, applies, layers)

    def positions(self, prompt_ids, sampling_params, layer=10, **gen_kwargs):
        """Sorted steered absolute positions for one layer."""
        _, by_layer = self.run(
            prompt_ids, sampling_params, layers=(layer,), **gen_kwargs
        )
        return sorted(p for p, _ in by_layer[layer])
