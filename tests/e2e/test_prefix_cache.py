# SPDX-License-Identifier: Apache-2.0
"""Steering-aware prefix caching: KV reuse keyed by config fingerprint.

Mechanism-level checks via RequestOutput.num_cached_tokens (no reliance
on output text, so no batch-shape numerics sensitivity):
- same prompt + same spec reuses blocks; a different scale or apply
  clause does not (the historic name-collision bug class);
- steered and unsteered requests never share blocks;
- length-sensitive specs (negative positions, generation windows) share
  only at equal prompt lengths;
- a prompt containing another request's generated tokens reuses only
  prompt-region blocks (phase boundary respected);
- capture and fresh runtime server-steering installs are rejected on a
  prefix-caching engine.
"""

import pytest

from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from helpers import DENSE_MODEL, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    enable_prefix_caching=True,
    enable_chunked_prefill=False,
    enforce_eager=True,
    gpu_memory_utilization=0.18,
    max_model_len=512,
)

PROMPT_48 = list(range(200, 248))


def run(llm, prompt_ids, steering=None, max_tokens=4):
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    return llm.generate(
        TokensPrompt(prompt_token_ids=list(prompt_ids)),
        sp,
        steering=steering,
        use_tqdm=False,
    )[0]


class TestFingerprintKeying:
    def test_same_spec_reuses_different_scale_does_not(self, llm):
        spec_a = steering_spec(scale=1.0)
        spec_b = steering_spec(scale=3.0)
        assert run(llm, PROMPT_48, spec_a).num_cached_tokens == 0
        assert run(llm, PROMPT_48, spec_a).num_cached_tokens > 0, (
            "identical spec must reuse its own blocks"
        )
        assert run(llm, PROMPT_48, spec_b).num_cached_tokens == 0, (
            "different scale must not reuse (fingerprint collision)"
        )
        assert run(llm, PROMPT_48, spec_b).num_cached_tokens > 0

    def test_different_apply_clause_does_not_reuse(self, llm):
        spec = steering_spec(scale=2.0)
        assert run(llm, PROMPT_48, spec).num_cached_tokens == 0
        assert run(llm, PROMPT_48, spec).num_cached_tokens > 0
        excl = steering_spec(scale=2.0, exclude_positions=[0])
        assert run(llm, PROMPT_48, excl).num_cached_tokens == 0, (
            "apply-clause change must re-key the blocks"
        )

    def test_steered_unsteered_isolation(self, llm):
        seeded = steering_spec(scale=1.5)
        run(llm, PROMPT_48, seeded)
        assert run(llm, PROMPT_48, None).num_cached_tokens == 0, (
            "unsteered request must not reuse steered blocks"
        )
        assert run(llm, PROMPT_48, None).num_cached_tokens > 0, (
            "unsteered requests must reuse unsteered blocks"
        )


class TestLengthSensitivity:
    def test_negative_position_keys_on_prompt_length(self, llm):
        spec = steering_spec(phases=["prompt"], positions=[-1])
        longer = PROMPT_48 + list(range(300, 308))
        assert run(llm, PROMPT_48, spec).num_cached_tokens == 0
        assert run(llm, longer, spec).num_cached_tokens == 0, (
            "longer prompt with a length-sensitive spec must not reuse"
        )
        assert run(llm, PROMPT_48, spec).num_cached_tokens > 0, (
            "equal prompt length must reuse"
        )

    def test_generation_window_keys_on_prompt_length(self, llm):
        spec = steering_spec(phases=["generation"], generation_window=(0, 2))
        longer = PROMPT_48 + list(range(400, 408))
        assert run(llm, PROMPT_48, spec).num_cached_tokens == 0
        assert run(llm, longer, spec).num_cached_tokens == 0
        assert run(llm, PROMPT_48, spec).num_cached_tokens > 0


class TestPhaseBoundary:
    def test_continuation_reuses_only_prompt_region(self, llm):
        spec = steering_spec(scale=0.5)
        out = run(llm, PROMPT_48, spec, max_tokens=20)
        continuation = PROMPT_48 + list(out.outputs[0].token_ids)
        cached = run(llm, continuation, spec).num_cached_tokens
        assert 0 < cached <= 48, (
            f"continuation must reuse only the 48-token prompt region, "
            f"got {cached}"
        )


class TestCachingEngineRejections:
    def test_capture_rejected(self, llm):
        with pytest.raises(Exception, match="(?i)prefix"):
            llm.llm_engine.collective_rpc("start_capture", args=("hidden_states",))

    def test_fresh_server_install_rejected(self, llm):
        from vllm.steer_vectors import to_engine_request

        with pytest.raises(Exception, match="(?i)salt"):
            llm.llm_engine.collective_rpc(
                "add_steer_vector",
                args=(to_engine_request(steering_spec(scale=1.0)),),
            )
