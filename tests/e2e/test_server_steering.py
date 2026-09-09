# SPDX-License-Identifier: Apache-2.0
"""Default and explicit requests share execution and prefix-cache identity.

These tests reuse the compiled capture engine. CPU tests cover request snapshots
across default updates; this suite checks actual cache separation and reuse.
"""

import pytest

from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from helpers import DENSE_MODEL, steering_spec
from test_capture_unified import ENGINE_KWARGS, ENGINE_PROFILE  # noqa: F401

PROMPT = list(range(400, 448))
PARAMS = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)


def test_invalid_steering_config_rejected_before_boot():
    from vllm.engine.arg_utils import EngineArgs

    with pytest.raises(ValueError, match="non-empty"):
        EngineArgs(
            model=DENSE_MODEL, steering_config='{"vectors": []}'
        ).create_engine_config()


def run(llm, **kwargs):
    return llm.generate(
        TokensPrompt(prompt_token_ids=list(PROMPT)), PARAMS,
        use_tqdm=False, **kwargs,
    )[0]


def test_default_override_and_off_share_request_cache_keys(llm):
    """Changing defaults preserves each effective configuration's cache."""
    first = steering_spec(scale=1.0, layers=[10])
    second = steering_spec(scale=2.0, layers=[10])
    assert llm.get_default_steering() == {"active": False}
    llm.set_default_steering(first)
    try:
        assert llm.get_default_steering()["active"]
        assert run(llm).num_cached_tokens == 0
        assert run(llm, steering=first).num_cached_tokens > 0

        prompts = [TokensPrompt(prompt_token_ids=list(PROMPT)) for _ in range(3)]
        mixed = llm.generate(
            prompts, PARAMS, steering=[None, False, second], use_tqdm=False,
        )
        assert mixed[0].num_cached_tokens > 0, "None must inherit the default"
        assert mixed[1].num_cached_tokens == 0, "False must use unsteered KV"
        assert mixed[2].num_cached_tokens == 0, "an override must have its own KV"
        assert run(llm, steering=False).num_cached_tokens > 0
        assert run(llm, steering=second).num_cached_tokens > 0

        llm.set_default_steering(second)
        assert run(llm).num_cached_tokens > 0, "the new default reuses explicit KV"
        assert run(llm, steering=first).num_cached_tokens > 0, (
            "updating a default must retain reusable KV for the previous config"
        )

        llm.set_default_steering(None)
        assert llm.get_default_steering() == {"active": False}
        assert run(llm).num_cached_tokens > 0, "clearing the default reuses off KV"
    finally:
        llm.set_default_steering(None)
