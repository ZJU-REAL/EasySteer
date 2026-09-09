# SPDX-License-Identifier: Apache-2.0
"""Public steering imports work with NumPy alone; heavy dependencies stay local."""

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TestPackageImports(unittest.TestCase):
    def run_script(self, script):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "vllm-steer"), str(ROOT)))}
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_accumulators_and_numpy_utilities_do_not_import_heavy_packages(self):
        self.run_script("""
            import importlib.abc
            import sys
            import numpy as np

            blocked = ("torch", "sklearn", "vllm", "gguf", "easysteer.steer.sae")
            class BlockHeavy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, *args):
                    if any(fullname == name or fullname.startswith(name + ".")
                           for name in blocked):
                        raise AssertionError("Unexpected dependency import: " + fullname)
            sys.meta_path.insert(0, BlockHeavy())

            import easysteer.steer as steer
            from easysteer.steer import (
                DiffMeanAccumulator, MomentsAccumulator, TopKCountAccumulator,
                StatisticalControlVector, extract_token_hiddens,
            )
            acc = MomentsAccumulator()
            acc.update(7, np.array([[1, 2], [3, 4]]))
            np.testing.assert_array_equal(acc.mean(7), [2, 3])
            assert steer.MomentsAccumulator is MomentsAccumulator
            assert "PCAExtractor" in dir(steer)
            assert not any(name == item or name.startswith(item + ".")
                           for name in sys.modules for item in blocked)
        """)

    def test_tensor_like_rows_detach_and_convert_to_cpu_float(self):
        import numpy as np

        from easysteer.steer import extract_token_hiddens
        from easysteer.steer.utils import extract_token_from_sequence
        calls = []
        class TensorRow:
            def detach(self):
                calls.append("detach")
                return self
            def cpu(self):
                calls.append("cpu")
                return self
            def float(self):
                calls.append("float")
                return self
            def numpy(self):
                calls.append("numpy")
                return np.array([2, 4], dtype=np.float32)
        positive, negative = extract_token_hiddens([[[TensorRow()]]], [0])
        np.testing.assert_array_equal(positive[0], [[2, 4]])
        assert negative == {}
        assert calls == ["detach", "cpu", "float", "numpy"]
        calls.clear()
        np.testing.assert_array_equal(
            extract_token_from_sequence([TensorRow()], "mean"), [2, 4])
        assert calls == ["detach", "cpu", "float", "numpy"]

    def test_sae_numpy_extraction_and_explicit_missing_torch_save_error(self):
        self.run_script("""
            import importlib.abc
            from pathlib import Path
            import sys
            import tempfile
            import types
            import numpy as np

            class NoTorch(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, *args):
                    if fullname == "torch" or fullname.startswith("torch."):
                        raise ModuleNotFoundError("torch unavailable", name="torch")
            sys.meta_path.insert(0, NoTorch())
            # This test exercises local NPZ extraction and never uses the network.
            sys.modules["requests"] = types.ModuleType("requests")
            from easysteer.steer import SAEFeatureExplorer
            assert "torch" not in sys.modules
            explorer = SAEFeatureExplorer(api_key="test")
            with tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "sae.npz"
                target = Path(folder) / "vector.pt"
                np.savez(source, W_dec=np.array([[1, 2], [3, 4]], dtype=np.float32))
                np.testing.assert_array_equal(explorer.extract_decoder_vector(source, 1), [3, 4])
                try:
                    explorer.extract_decoder_vector(source, 1, save_path=target)
                except ImportError as exc:
                    assert "requires torch" in str(exc)
                else:
                    raise AssertionError("Missing torch must fail the requested save")
                assert not target.exists()
                assert not target.with_suffix(".npy").exists()
        """)

    def test_capture_import_and_selection_do_not_load_steering(self):
        self.run_script("""
            import importlib.abc
            import sys
            class NoSteering(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, *args):
                    if fullname.startswith(("vllm.steer_vectors", "vllm.model_hooks.steering")):
                        raise AssertionError("capture depends on steering: " + fullname)
            sys.meta_path.insert(0, NoSteering())
            from vllm.capture import SelectSpec, StreamConfig, deserialize_captured
            from vllm.model_hooks.capture.policy import CaptureRequestPolicy
            spec = SelectSpec(generation="all")
            config = StreamConfig(select=spec.to_wire(), dtype="float16")
            policy = CaptureRequestPolicy()
            policy.record_rpc("start_capture", ("hidden_states",), {"select": spec.to_wire()})
            assert not policy.skip_prefix_read([1, 2], None)
            assert deserialize_captured({}) == ({}, {})
        """)

    def test_public_steering_api_does_not_load_runtime(self):
        self.run_script("""
            import importlib.abc
            import sys
            class NoRuntime(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, *args):
                    runtime_modules = (
                        "vllm.model_hooks.steering.controllers",
                        "vllm.model_hooks.steering.algorithms",
                        "vllm.model_hooks.steering.graph",
                        "vllm.model_hooks.steering.worker_manager",
                        "vllm.model_hooks.steering.payload_cache",
                        "vllm.model_hooks.capture",
                        "vllm.model_hooks.selection.runtime",
                    )
                    if any(fullname == name or fullname.startswith(name + ".")
                           for name in runtime_modules):
                        raise AssertionError("authoring loads runtime: " + fullname)
            sys.meta_path.insert(0, NoRuntime())
            from vllm.steer_vectors import ApplySpec, DirectionVector, SteeringSpec, VectorSpec
            from vllm.steer_vectors.api import VectorSpec as ApiVectorSpec
            from vllm.steer_vectors.payloads import DirectionVector as PublicPayload
            assert ApiVectorSpec is VectorSpec and PublicPayload is DirectionVector
            spec = SteeringSpec(vectors=[VectorSpec(
                data=DirectionVector({0: [1., 2.]}), apply=ApplySpec(prompt="all"),
            )])
            assert spec.vectors[0].algorithm == "direct"
        """)


if __name__ == "__main__":
    unittest.main()
