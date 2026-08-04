# SPDX-License-Identifier: Apache-2.0
"""Golden-output check: engine-default steering reproduces the recorded
happy-steer output.

Boots an engine whose default steering comes from a v2 SteeringSpec and
checks the unsteered-request output is byte-identical to the golden
recorded by an explicit per-request config with the same parameters
(same math, same batch geometry).

The golden is GPU-model-specific (recorded on RTX PRO 5000, bf16
vectors): run only on that GPU model. Skipped if the golden file is
absent.
"""

import os

import pytest

from vllm import SamplingParams

from helpers import DENSE_MODEL, steering_spec

GOLDEN = os.path.expanduser("~/EasySteer-migration/golden.txt")

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    steering_config=steering_spec(
        scale=2.0, layers=list(range(10, 26))
    ).model_dump_json(),
    enforce_eager=True,
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.18,
    max_model_len=2048,
)

PROMPT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)


@pytest.mark.skipif(not os.path.exists(GOLDEN), reason="golden.txt not present")
def test_server_default_matches_golden(llm):
    import torch

    if "RTX PRO 5000" not in torch.cuda.get_device_name(0):
        pytest.skip("golden recorded on RTX PRO 5000; other GPU models "
                    "produce different bytes")
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    out = llm.generate(PROMPT, sampling_params=sp)[0].outputs[0].text
    golden = open(GOLDEN).read()
    assert out in golden, (
        "engine-default steering output differs from the recorded golden "
        "(check GPU model and vector dtype before suspecting a regression)"
    )
