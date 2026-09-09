"""CPU checks for test process isolation and suite selection; no vLLM import."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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
                       RUNNER_PROBE_LOG=str(calls), STEER_TEST_GPU_PAUSE="0")
            env.pop("STEER_TEST_RESULTS_DIR", None)
            env.pop("STEER_TEST_QWEN3", None)
            for group, count in [("cpu", 1), ("kernels", 1), ("baseline", 9), ("moe-core", 3),
                                 ("extended", 1)]:
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
                    self.assertEqual(len(invocations), count)
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


if __name__ == "__main__":
    unittest.main()
