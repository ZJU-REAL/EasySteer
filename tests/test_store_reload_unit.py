# SPDX-License-Identifier: Apache-2.0
"""CPU unit test: VectorStore staleness handling + fingerprint versioning.

Verifies that a vector file regenerated at the same path loads fresh
(instead of serving the stale cached payload), that identical files still
dedup, that stale versions don't hold store capacity, and that the config
fingerprint changes with the file version (so live slots aren't reused
across versions).
"""

import os
import tempfile
import time
import types
from unittest import mock

from vllm.steer_vectors.store import VectorStore
from vllm.steer_vectors.worker_manager import config_fingerprint
from vllm.steer_vectors.request import SteerVectorRequest

failures = []


def check(name, cond):
    if not cond:
        failures.append(name)
        print("FAIL:", name)


cfg = types.SimpleNamespace(max_steer_vectors=8)
loads = []


def fake_load(**kwargs):
    loads.append(kwargs["steer_vector_model_path"])
    return types.SimpleNamespace(tag=len(loads), layer_payloads={})


with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "vec.gguf")
    with open(path, "wb") as f:
        f.write(b"version-1")

    with mock.patch(
        "vllm.steer_vectors.models.SteerVectorModel.from_local_checkpoint",
        side_effect=fake_load,
    ):
        store = VectorStore("cpu", cfg)
        a = store.get(path, "direct")
        b = store.get(path, "direct")
        check("identical file dedups", a is b and len(loads) == 1)

        time.sleep(0.01)
        with open(path, "wb") as f:
            f.write(b"version-2-different-length")
        c = store.get(path, "direct")
        check("rewritten file reloads", c is not a and len(loads) == 2)
        check("stale version purged", len(store._entries) == 1)

        d = store.get(path, "direct")
        check("new version cached", d is c and len(loads) == 2)

        store.reload(path, "direct")
        check("explicit reload forces load", len(loads) == 3)

    def req():
        return SteerVectorRequest(
            "r", 1, steer_vector_local_path=path, scale=1.0,
            target_layers=[0],
            prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])

    fp1 = config_fingerprint(req())
    time.sleep(0.01)
    with open(path, "wb") as f:
        f.write(b"version-3-yet-another-length!")
    fp2 = config_fingerprint(req())
    check("fingerprint tracks file version", fp1 != fp2)
    check("fingerprint stable for same version",
          fp2 == config_fingerprint(req()))

print("OVERALL:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
