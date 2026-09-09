# SPDX-License-Identifier: Apache-2.0
"""Optional, environment-specific text comparison for engine-default steering.

Run with an explicit STEER_TEST_GOLDEN JSON record; see tests/README.md.
The record's environment and workload metadata must match before the engine
is started. Text equality is a diagnostic for that recorded setup, not a
portable guarantee of deterministic vLLM output.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from helpers import DENSE_MODEL, DENSE_VECTOR, steering_spec

SPEC = steering_spec(scale=2.0, layers=list(range(10, 26)))
ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    steering_config=SPEC.model_dump_json(),
    enforce_eager=True,
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    async_scheduling=False,
    dtype="bfloat16",
    gpu_memory_utilization=0.18,
    max_model_len=512,
    max_num_batched_tokens=512,
    max_num_seqs=1,
)

PROMPT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
SAMPLING = dict(temperature=0.0, max_tokens=128, ignore_eos=True, seed=0)


@pytest.fixture(scope="module")
def golden_record():
    import torch
    import vllm

    path = os.environ.get("STEER_TEST_GOLDEN")
    if not path:
        pytest.skip("set STEER_TEST_GOLDEN to opt into a recorded text comparison")
    record = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    expected = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": DENSE_MODEL,
        "dtype": "bfloat16",
        "vector_sha256": hashlib.sha256(Path(DENSE_VECTOR).read_bytes()).hexdigest(),
        "steering": SPEC.model_dump(
            mode="json", exclude={"vectors": {"__all__": {"source"}}}
        ),
        "prompt": PROMPT,
        "sampling": SAMPLING,
        "engine": {
            key: value
            for key, value in ENGINE_KWARGS.items()
            if key not in {"model", "steering_config", "dtype"}
        },
    }
    assert record.get("metadata") == expected, (
        "golden metadata does not match this environment/workload; "
        "compare only against a separately recorded reference with matching "
        f"metadata. Expected metadata: {json.dumps(expected, indent=2)}"
    )
    assert isinstance(record.get("output_text"), str) and record["output_text"], (
        "golden output_text must contain one complete recorded completion"
    )
    return record


def test_server_default_matches_recorded_text(golden_record, request):
    from vllm import SamplingParams

    # Request the engine only after validating the record, before allocating VRAM.
    llm = request.getfixturevalue("llm")
    output = llm.generate(PROMPT, sampling_params=SamplingParams(**SAMPLING))[0]
    assert output.outputs[0].text == golden_record["output_text"], (
        "output differs from the recorded text; repeat with matching controls "
        "and inspect steering traces before classifying a regression"
    )
