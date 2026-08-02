# SPDX-License-Identifier: Apache-2.0
"""VectorStore staleness handling + fingerprint file versioning.

A vector file regenerated at the same path must load fresh (not serve
the stale cached payload), identical files must dedup, stale versions
must not hold store capacity, and the config fingerprint must change
with the file version so live slots aren't reused across versions.
"""

import os
import time
import types
from unittest import mock

import pytest

from vllm.steer_vectors.request import SteerVectorRequest
from vllm.steer_vectors.store import VectorStore
from vllm.steer_vectors.worker_manager import config_fingerprint


@pytest.fixture()
def vec_path(tmp_path):
    path = os.path.join(tmp_path, "vec.gguf")
    with open(path, "wb") as f:
        f.write(b"version-1")
    return path


def _rewrite(path, content):
    time.sleep(0.01)
    with open(path, "wb") as f:
        f.write(content)


def test_store_versioning(vec_path):
    cfg = types.SimpleNamespace(max_steer_vectors=8)
    loads = []

    def fake_load(**kwargs):
        loads.append(kwargs["steer_vector_model_path"])
        return types.SimpleNamespace(tag=len(loads), layer_payloads={})

    with mock.patch(
        "vllm.steer_vectors.models.LoadedSteerVector.from_local_checkpoint",
        side_effect=fake_load,
    ):
        store = VectorStore("cpu", cfg)
        a = store.get(vec_path, "direct")
        b = store.get(vec_path, "direct")
        assert a is b and len(loads) == 1, "identical file must dedup"

        _rewrite(vec_path, b"version-2-different-length")
        c = store.get(vec_path, "direct")
        assert c is not a and len(loads) == 2, "rewritten file must reload"
        assert len(store._entries) == 1, "stale version must be purged"

        d = store.get(vec_path, "direct")
        assert d is c and len(loads) == 2, "new version must be cached"

        store.reload(vec_path, "direct")
        assert len(loads) == 3, "explicit reload must force a load"


def test_fingerprint_tracks_file_version(vec_path):
    def req():
        return SteerVectorRequest(
            "r", 1, steer_vector_local_path=vec_path, scale=1.0,
            target_layers=[0],
            apply_spec={"phases": ["prompt", "generation"]})

    fp1 = config_fingerprint(req())
    _rewrite(vec_path, b"version-3-yet-another-length!")
    fp2 = config_fingerprint(req())
    assert fp1 != fp2, "fingerprint must track the file version"
    assert fp2 == config_fingerprint(req()), (
        "fingerprint must be stable for the same version"
    )
