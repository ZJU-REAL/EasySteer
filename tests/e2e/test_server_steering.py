# SPDX-License-Identifier: Apache-2.0
"""Engine-default steering (--steering-config) with prefix caching.

One eager engine whose default steering comes from a v2 SteeringSpec
JSON, with prefix caching enabled (salt + reset pattern). Covers:
- the engine boots from the spec and the default slot steers every
  computed token (trace oracle);
- prefix-cache reuse under the server salt (cold then warm);
- runtime spec replacement (worker RPC + reset_prefix_cache, mirroring
  POST /v1/steering): next request is cold, then warm again;
- an invalid steering_config is rejected at engine construction
  (checked without booting: config validation runs before workers).
"""

import pytest

from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from helpers import DENSE_MODEL, DENSE_VECTOR, steering_spec

SPEC = steering_spec(scale=1.0, layers=[10])

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    steering_config=SPEC.model_dump_json(),
    enable_prefix_caching=True,
    enable_chunked_prefill=False,
    enforce_eager=True,
    gpu_memory_utilization=0.18,
    max_model_len=512,
)

PROMPT = list(range(400, 448))
PARAMS = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)


def test_invalid_steering_config_rejected_before_boot():
    from vllm.engine.arg_utils import EngineArgs

    with pytest.raises(Exception, match="non-empty"):
        EngineArgs(
            model=DENSE_MODEL, steering_config='{"vectors": []}'
        ).create_engine_config()


def run(llm):
    return llm.generate(
        TokensPrompt(prompt_token_ids=list(PROMPT)), PARAMS, use_tqdm=False
    )[0]


def test_default_slot_steers_all_tokens(trace):
    out, by_layer = trace.run(PROMPT, PARAMS, layers=(10,))
    positions = {p for p, _ in by_layer[10]}
    assert positions == set(range(48 + 3)), (
        "engine-default steering must cover every computed token"
    )
    assert out.num_cached_tokens == 0


def test_warm_reuse_under_server_salt(trace):
    out, by_layer = trace.run(PROMPT, PARAMS, layers=(10,))
    assert 0 < out.num_cached_tokens <= 48
    decode_positions = {p for p, _ in by_layer[10]}
    assert {48, 49, 50} <= decode_positions, (
        "warm run must still steer the computed suffix and decode tokens"
    )


def test_runtime_spec_replacement_resets_cache(llm):
    from vllm.steer_vectors import to_engine_request

    new_spec = steering_spec(scale=2.0, layers=[10])
    llm.llm_engine.collective_rpc(
        "add_steer_vector",
        args=(to_engine_request(new_spec, name="__server__", int_id=1),),
    )
    llm.reset_prefix_cache()

    assert run(llm).num_cached_tokens == 0, (
        "post-replacement request must be cold (old-config blocks dropped)"
    )
    assert run(llm).num_cached_tokens > 0, (
        "new-config blocks must be reusable"
    )
