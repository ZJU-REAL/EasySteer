# SPDX-License-Identifier: Apache-2.0
"""--steer-require-preload frontend enforcement.

With require_preload set, steering specs referencing vectors that were
not explicitly preloaded are rejected with a clear VLLMClientError at the
frontend — the engine stays alive — and succeed (and actually steer)
after LLM.preload_steer_vectors.
"""

import os

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMClientError

from helpers import DENSE_MODEL, DENSE_VECTOR, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_algorithms=["direct"],
    steer_multi_vector=True,
    steer_require_preload=True,
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.18,
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=32,
)

PROMPT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
LAYERS = list(range(10, 26))
SP = SamplingParams(temperature=0.0, max_tokens=48)


def test_unpreloaded_vector_rejected_then_accepted(llm):
    spec = steering_spec(scale=2.0, layers=LAYERS)

    with pytest.raises(VLLMClientError, match="not preloaded"):
        llm.generate(PROMPT, steering=spec, sampling_params=SP)

    plain = llm.generate(PROMPT, sampling_params=SP)[0].outputs[0].text

    llm.preload_steer_vectors([DENSE_VECTOR])
    steered = llm.generate(
        PROMPT, steering=spec, sampling_params=SP
    )[0].outputs[0].text
    assert steered != plain, "preloaded spec must actually steer"


def test_unpreloaded_multi_vector_rejected(llm, tmp_path):
    import gguf
    import numpy as np
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    width = llm.llm_engine.vllm_config.model_config.hf_text_config.hidden_size
    paths = []
    for layer in (10, 12):
        path = tmp_path / f"direction-{layer}.gguf"
        writer = gguf.GGUFWriter(str(path), "steervector")
        writer.add_tensor(
            f"direction.{layer}", np.full(width, layer / 1000, dtype=np.float32)
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        paths.append(str(path))
    llm.preload_steer_vectors([paths[0]])
    spec = SteeringSpec(vectors=[
        VectorSpec(source=path, layers=[layer],
                   apply=ApplySpec(prompt="all", generation="all"))
        for path, layer in zip(paths, (10, 12))
    ])
    with pytest.raises(VLLMClientError, match="not preloaded") as caught:
        llm.generate(PROMPT, steering=spec, sampling_params=SP)
    assert paths[1] in str(caught.value)
    assert paths[0] not in str(caught.value)
    llm.preload_steer_vectors([paths[1]])
    output = llm.generate(
        PROMPT, steering=spec, use_tqdm=False,
        sampling_params=SamplingParams(max_tokens=2, ignore_eos=True),
    )[0].outputs[0]
    assert output.finish_reason == "length" and len(output.token_ids) == 2
