"""CPU checks for test process isolation and suite selection; no vLLM import."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from helpers import CaptureGraphWorkerExtension, TraceOracle, read_trace, trace_cursor


TESTS = Path(__file__).resolve().parents[1]
PROCESS_RUNNER = TESTS / "run_process.py"
spec = importlib.util.spec_from_file_location("test_process_runner", PROCESS_RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestIsolatedRunner(unittest.TestCase):
    def test_exit_status_and_timing_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            process = subprocess.run(
                [sys.executable, str(PROCESS_RUNNER), "--result", str(result),
                 "--", sys.executable, "-c", "raise SystemExit(3)"],
                capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(process.returncode, 3)
            record = json.loads(result.read_text())
            self.assertEqual(record["exit_code"], 3)
            self.assertEqual(record["remaining_pids"], [])
            self.assertGreaterEqual(record["seconds"], record["cleanup_seconds"])

    def test_cleanup_targets_only_its_own_children(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                record_path = Path(directory) / "child.json"
                child_code = (
                    "import json, os, subprocess, sys; "
                    "p = subprocess.Popen([sys.executable, '-c', "
                    "'import time; time.sleep(30)']); "
                    "open(sys.argv[1], 'w').write(json.dumps("
                    "{'child': p.pid, 'group': os.getpgrp()}))"
                )
                result = subprocess.run(
                    [sys.executable, str(PROCESS_RUNNER), "--cleanup-timeout", "2",
                     "--", sys.executable, "-c", child_code, str(record_path)],
                    capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                record = json.loads(record_path.read_text())
                self.assertEqual(runner.live_group_members(record["group"]), [])
                self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_groups_and_shared_engine_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = root / "calls.jsonl"
            fake_python = root / "python"
            fake_python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['RUNNER_PROBE_LOG'], 'a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            fake_python.chmod(0o755)
            env = dict(os.environ, STEER_TEST_PYTHON=str(fake_python),
                       STEER_TEST_MOE_MODEL=directory, STEER_TEST_MODEL=directory,
                       STEER_TEST_QWEN3=directory,
                       RUNNER_PROBE_LOG=str(calls), STEER_TEST_GPU_PAUSE="0")
            env.pop("STEER_TEST_RESULTS_DIR", None)
            for group, count in [("cpu", 1), ("kernels", 2), ("baseline", 10),
                                 ("moe-core", 3), ("extended", 1), ("all", None)]:
                with self.subTest(group=group):
                    calls.write_text("")
                    run_env = dict(env)
                    if group == "kernels":
                        run_env.pop("STEER_TEST_MODEL", None)
                        run_env.pop("STEER_TEST_MOE_MODEL", None)
                    result = subprocess.run(
                        ["bash", str(TESTS / "run_suites.sh"), group],
                        env=run_env, capture_output=True, text=True, timeout=20,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    invocations = [json.loads(line) for line in calls.read_text().splitlines()]
                    if count is not None:
                        self.assertEqual(len(invocations), count)
                    selected = {
                        arg.split("::", 1)[0]
                        for invocation in invocations
                        for arg in invocation
                        if arg.startswith(("kernels/", "e2e/", "moe/"))
                    }
                    kernels = {
                        str(path.relative_to(TESTS))
                        for path in (TESTS / "kernels").glob("test_*.py")
                    }
                    if group in {"kernels", "baseline", "all"}:
                        self.assertLessEqual(kernels, selected)
                    if group == "all":
                        maintained_gpu = {
                            str(path.relative_to(TESTS))
                            for folder in ("kernels", "e2e", "moe")
                            for path in (TESTS / folder).glob("test_*.py")
                        }
                        # Recorded hardware-specific text is an explicit opt-in.
                        maintained_gpu.remove("e2e/test_golden_sentiment.py")
                        self.assertEqual(selected, maintained_gpu)
                    if group == "moe-core":
                        self.assertIn("moe/test_moe.py", invocations[0])
                        self.assertIn("moe/test_steermoe.py", invocations[0])
                    if group == "extended":
                        self.assertIn("--steer-extended", invocations[0])
            help_result = subprocess.run(
                ["bash", str(TESTS / "run_suites.sh"), "--help"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(help_result.returncode, 0)
            self.assertIn("moe-core", help_result.stdout)

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "summary.tsv"
            record.write_text("previous run\n")
            result = subprocess.run(
                ["bash", str(TESTS / "run_suites.sh"), "cpu"],
                env=dict(os.environ, STEER_TEST_PYTHON=sys.executable,
                         STEER_TEST_RESULTS_DIR=directory),
                capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(record.read_text(), "previous run\n")


class TestTraceOracle(unittest.TestCase):
    """Trace regressions need neither Torch nor a model engine."""

    @staticmethod
    def write_worker(path, step=1, positions=(0, 1, 2), layer=10):
        records = [{
            "type": "step", "step": step, "req_ids": ["request"],
            "query_start_loc": [0, 3], "num_computed": [0], "num_output": [0],
        }]
        if positions is not None:
            records.append({
                "type": "apply", "step": step, "layer": layer,
                "positions": positions,
            })
        with path.open("a") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    @staticmethod
    def oracle(directory, tp=2):
        config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp))
        return TraceOracle(SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config)),
                           directory)

    def test_new_worker_steps_are_not_hidden_by_old_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_worker(root / "old.jsonl", step=100)
            self.write_worker(root / "current.jsonl", step=1)
            cursor = trace_cursor(directory)
            self.write_worker(root / "current.jsonl", step=2)
            self.write_worker(root / "new.jsonl", step=1)
            steps, applies = read_trace(directory, cursor, (10,))
            self.assertEqual(set(steps), {("current.jsonl", 2), ("new.jsonl", 1)})
            self.assertEqual(len(applies), 2)

    def test_tp_workers_keep_independent_geometry_and_exact_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in range(2):
                self.write_worker(root / f"worker{rank}.jsonl")
            steps, applies = read_trace(directory, 0, (10,))
            oracle = self.oracle(directory)
            self.assertEqual(len(steps), 2)
            self.assertEqual(oracle._positions_by_worker(steps, applies, (10,)),
                             {10: [(0, True), (1, True), (2, True)]})
            self.assertEqual(len(oracle.last_by_worker), 2)
            # A peer with different geometry must not be joined to rank zero.
            steps[("worker1.jsonl", 1)]["num_computed"] = [5]
            with self.assertRaises(AssertionError):
                oracle._positions_by_worker(steps, applies, (10,))

    def test_missing_peer_and_missing_peer_applications_fail(self):
        for missing_worker in (True, False):
            with self.subTest(missing_worker=missing_worker):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.write_worker(root / "worker0.jsonl")
                    if not missing_worker:
                        self.write_worker(root / "worker1.jsonl", positions=None)
                    steps, applies = read_trace(directory, 0, (10,))
                    with self.assertRaises(AssertionError):
                        self.oracle(directory)._positions_by_worker(steps, applies, (10,))

    def test_only_explicit_off_allows_absent_trace_and_rejects_any_apply(self):
        cases = (
            ("implicit missing", {}, (), False),
            ("steered missing", {"steering": object()}, (), False),
            ("off missing", {"steering": False}, (), True),
            ("off geometry", {"steering": False}, ((None, 10),) * 2, True),
            ("off missing peer", {"steering": False}, ((None, 10),), False),
            ("off apply", {"steering": False}, (((0,), 10),) * 2, False),
            ("off other layer", {"steering": False}, (((0,), 12),) * 2, False),
        )
        inputs = ModuleType("vllm.inputs")
        inputs.TokensPrompt = dict
        for name, kwargs, records, succeeds in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                oracle = self.oracle(directory)
                # A no-trace run must not retain an earlier worker observation.
                oracle.last_by_worker = {"previous": {10: [(0, True)]}}

                def generate(prompt, sampling_params, **passed):
                    self.assertEqual(passed, dict(kwargs, use_tqdm=False))
                    for rank, (positions, layer) in enumerate(records):
                        self.write_worker(
                            Path(directory) / f"worker{rank}.jsonl",
                            positions=positions, layer=layer,
                        )
                    return ["output"]

                oracle.llm.generate = generate
                with patch.dict(sys.modules, {"vllm.inputs": inputs}):
                    if succeeds:
                        self.assertEqual(
                            oracle.run([100], None, **kwargs),
                            ("output", {10: []}),
                        )
                        self.assertEqual(len(oracle.last_by_worker), len(records))
                    else:
                        with self.assertRaises(AssertionError):
                            oracle.run([100], None, **kwargs)
                        self.assertEqual(oracle.last_by_worker, {})


class TestGraphReplayProbe(unittest.TestCase):
    def test_native_replays_classify_and_restore_cached_breakable_methods(self):
        calls = []

        class Graph:
            def replay(self):
                calls.append(self)

        full, piecewise, cached, lazy, capture_only = [Graph() for _ in range(5)]
        def eager():
            calls.append("eager")

        cached_segments = [cached.replay, eager]
        entries = {
            0: SimpleNamespace(capture=SimpleNamespace(segments=cached_segments)),
        }
        mode = SimpleNamespace(PIECEWISE="piecewise")
        modules = {name: ModuleType(name) for name in (
            "torch", "vllm", "vllm.compilation", "vllm.config",
            "vllm.compilation.cuda_graph", "vllm.compilation.breakable_cudagraph",
            "vllm.config.compilation",
        )}
        modules["torch"].cuda = SimpleNamespace(CUDAGraph=Graph)
        modules["vllm.config.compilation"].CUDAGraphMode = mode
        modules["vllm.compilation.cuda_graph"].CUDAGraphWrapper = SimpleNamespace(
            _all_instances=[SimpleNamespace(
                runtime_mode=mode.PIECEWISE,
                concrete_cudagraph_entries={0: SimpleNamespace(cudagraph=piecewise)},
            )],
        )
        modules["vllm.compilation.breakable_cudagraph"].BreakableCUDAGraphWrapper = (
            SimpleNamespace(_all_instances=[SimpleNamespace(entries=entries)])
        )
        worker = CaptureGraphWorkerExtension()
        worker.model_runner = SimpleNamespace(
            cudagraph_manager=SimpleNamespace(graphs={0: full}),
            capture_graph_manager=SimpleNamespace(graphs={0: capture_only}),
            capture_status=lambda stream: {"topology": {"tp_rank": 0}},
        )
        original = Graph.replay
        with patch.dict(sys.modules, modules):
            try:
                worker.graph_test_start()
                full.replay()
                full.replay()
                piecewise.replay()
                for segment in cached_segments:
                    segment()
                lazy_segments = [lazy.replay]
                entries[1] = SimpleNamespace(
                    capture=SimpleNamespace(segments=lazy_segments)
                )
                lazy_segments[0]()
                capture_only.replay()
                self.assertEqual(worker.graph_test_read(), {
                    "rank": 0, "full_replays": 2, "piecewise_replays": 3,
                })
                self.assertEqual(calls.count("eager"), 1)
            finally:
                worker.graph_test_stop()
            self.assertIs(Graph.replay, original)
            self.assertIs(cached_segments[0].__func__, original)
            self.assertIs(lazy_segments[0].__func__, original)
            self.assertIs(cached_segments[1], eager)
            cached_segments[0]()
            lazy_segments[0]()
            self.assertEqual(calls.count(cached), 2)
            self.assertEqual(calls.count(lazy), 2)
            self.assertTrue(worker.graph_test_stop())


if __name__ == "__main__":
    unittest.main()
