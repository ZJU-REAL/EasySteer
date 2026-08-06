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
        "vllm.steer_vectors.controller_manager.LoadedSteerVector.from_local_checkpoint",
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




def test_store_forwards_target_layers(vec_path):
    """Single-layer formats need target_layers at load; the store must
    forward them and key entries per layer set (correctness over dedup)."""
    cfg = types.SimpleNamespace(max_steer_vectors=8)
    seen_layers = []

    def fake_load(**kwargs):
        seen_layers.append(kwargs["target_layers"])
        return types.SimpleNamespace(layer_payloads={})

    with mock.patch(
        "vllm.steer_vectors.controller_manager.LoadedSteerVector.from_local_checkpoint",
        side_effect=fake_load,
    ):
        store = VectorStore("cpu", cfg)
        a = store.get(vec_path, "lm_steer", target_layers=[11])
        b = store.get(vec_path, "lm_steer", target_layers=[11])
        c = store.get(vec_path, "lm_steer", target_layers=[3, 5])
        assert a is b, "same layer set must dedup"
        assert c is not a, "different layer set must load its own entry"
        assert seen_layers == [[11], [3, 5]]


def test_fingerprint_tracks_file_version(vec_path):
    def req():
        return SteerVectorRequest(
            "r", 1, steer_vector_local_path=vec_path, scale=1.0,
            target_layers=[0],
            apply_spec={"prompt": "all", "generation": "all"})

    fp1 = config_fingerprint(req())
    _rewrite(vec_path, b"version-3-yet-another-length!")
    fp2 = config_fingerprint(req())
    assert fp1 != fp2, "fingerprint must track the file version"
    assert fp2 == config_fingerprint(req()), (
        "fingerprint must be stable for the same version"
    )

def test_store_inline_payloads_dedup_by_hash():
    import numpy as np
    import torch

    from vllm.steer_vectors.payloads import DirectionVector

    cfg = types.SimpleNamespace(max_steer_vectors=8, adapter_dtype=torch.float32)
    store = VectorStore("cpu", cfg)
    wire = DirectionVector({3: np.ones(4)}).to_wire()
    a = store.get_inline(wire, "direct")
    b = store.get_inline(DirectionVector({3: np.ones(4)}).to_wire(), "direct")
    c = store.get_inline(DirectionVector({3: np.ones(4) * 2}).to_wire(), "direct")
    assert a is b, "byte-identical payloads must share one entry"
    assert c is not a, "different payload content must load its own entry"
    assert set(a.layer_payloads) == {3}
