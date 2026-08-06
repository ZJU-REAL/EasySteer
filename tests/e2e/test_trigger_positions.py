# SPDX-License-Identifier: Apache-2.0
"""Trigger-position accuracy on a compiled, prefix-caching engine.

The eager trace oracle (test_apply_semantics) pins exact positions;
this module closes the two gaps it leaves: position-sensitive specs
under in-graph steering (compiled execution, per-token row tables) and
under prefix caching, where a warm request's steered rows fall inside
the KV-cache hit and are never recomputed — correctness there rests on
the config-fingerprinted block keying, which is verified here end to
end as byte parity between cold (reset cache) and warm runs of the
same request. Differential checks guard against vacuous parity: every
spec must actually change the output, and steering different positions
must produce different outputs.

Covers absolute head positions (deep inside the cached region),
negative tail positions, position ranges, generation windows, and a
window spanning both prefill and generation.
"""

import pytest

from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from helpers import DENSE_MODEL, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_algorithms=["direct"],
    enforce_eager=False,
    enable_prefix_caching=True,
    enable_chunked_prefill=False,
    gpu_memory_utilization=0.18,
    max_model_len=512,
)

PROMPT = list(range(200, 248))  # 48 tokens: 3 full blocks of 16
LAYERS = tuple(range(10, 26))
SCALE = 20.0


def spec(**kwargs):
    kwargs.setdefault("scale", SCALE)
    kwargs.setdefault("layers", LAYERS)
    return steering_spec(**kwargs)


POSITION_SPECS = {
    "tail_negative": dict(phases=["prompt"], positions=[-1]),
    "head_absolute": dict(phases=["prompt"], positions=[0, 1, 2, 3]),
    "prompt_range": dict(phases=["prompt"], positions=[-4, -3, -2, -1]),
    "generation_window": dict(phases=["generation"], generation_window=(0, 2)),
    "cross_phase": dict(phases=["prompt", "generation"], positions=[-2, -1],
                        generation_window=(0, 2)),
    "prompt_window_tail": dict(phases=["prompt"], prompt_window=(-4, None)),
    "generation_positions": dict(phases=["generation"],
                                 generation_positions=[0, 1]),
    "exclude_twins": dict(phases=["prompt"], prompt_window=(-6, None),
                          exclude_prompt_window=(-4, -2)),
}


def run(llm, prompt_ids, steering=None, max_tokens=24):
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    return llm.generate(
        TokensPrompt(prompt_token_ids=list(prompt_ids)),
        sp,
        steering=steering,
        use_tqdm=False,
    )[0]


def test_declaration_resolves_in_graph(llm):
    """The direct-only declaration must keep full CUDA graphs — the
    module is about in-graph steering, not split mode."""
    cfg = llm.llm_engine.vllm_config
    assert cfg.steer_vector_config.graph_mode == "in_graph"
    assert cfg.compilation_config.cudagraph_mode.has_full_cudagraphs()


class TestWarmColdParity:
    """Cold (reset cache) and warm (cache hit) runs of an identical
    position-sensitive request must be byte-identical: warm requests
    reuse KV computed under the same steering fingerprint, so steered
    rows inside the cached region need no recomputation."""

    @pytest.mark.parametrize("name", sorted(POSITION_SPECS))
    def test_parity_and_effect(self, llm, name):
        s = spec(**POSITION_SPECS[name])
        llm.reset_prefix_cache()
        unsteered = run(llm, PROMPT).outputs[0].text
        llm.reset_prefix_cache()
        cold = run(llm, PROMPT, s)
        assert cold.num_cached_tokens == 0, "cold run must start uncached"
        assert cold.outputs[0].text != unsteered, (
            f"{name}: spec produced no effect; parity would be vacuous"
        )
        warm = run(llm, PROMPT, s)
        assert warm.num_cached_tokens > 0, "warm run must hit the cache"
        assert warm.outputs[0].text == cold.outputs[0].text, (
            f"{name}: warm output diverged from cold — steered rows in "
            f"the cached region were not equivalent to recomputation"
        )


class TestPartialPrefixHit:
    def test_head_positions_survive_partial_hit(self, llm):
        """A longer prompt sharing a steered head reuses only the head
        blocks; the recomputed tail plus cached steered head must equal
        the fully-cold run of the same request."""
        s = spec(phases=["prompt"], positions=[0, 1, 2, 3])
        longer = PROMPT + list(range(300, 316))
        llm.reset_prefix_cache()
        cold_long = run(llm, longer, s)
        assert cold_long.num_cached_tokens == 0
        llm.reset_prefix_cache()
        run(llm, PROMPT, s)  # fills the steered head blocks
        partial = run(llm, longer, s)
        assert 0 < partial.num_cached_tokens <= len(PROMPT), (
            f"expected a partial head hit, got "
            f"{partial.num_cached_tokens} cached tokens"
        )
        assert partial.outputs[0].text == cold_long.outputs[0].text, (
            "partial-hit output diverged from the fully-cold run"
        )


class TestPositionsAreLoadBearing:
    """Different trigger positions must produce different outputs —
    detects steering that fires but at the wrong rows."""

    def test_head_vs_tail_differ(self, llm):
        llm.reset_prefix_cache()
        head = run(llm, PROMPT, spec(phases=["prompt"], positions=[0]))
        tail = run(llm, PROMPT, spec(phases=["prompt"], positions=[-1]))
        assert head.outputs[0].text != tail.outputs[0].text

    def test_window_offset_shifts_first_steered_step(self, llm):
        """A (2, 4) window leaves decode steps 0-1 unsteered: its first
        two generated tokens must match the unsteered run, while a
        (0, 2) window's must not."""
        llm.reset_prefix_cache()
        unsteered = run(llm, PROMPT)
        early = run(llm, PROMPT, spec(phases=["generation"],
                                      generation_window=(0, 2)))
        late = run(llm, PROMPT, spec(phases=["generation"],
                                     generation_window=(2, 4)))
        base_head = list(unsteered.outputs[0].token_ids[:2])
        assert list(late.outputs[0].token_ids[:2]) == base_head, (
            "steps before the window opening must be unsteered"
        )
        assert list(early.outputs[0].token_ids[:2]) != base_head, (
            "steps inside the window must be steered"
        )


class TestCoBatchedPerRequestPositions:
    """Continuous-batching adversarial check on the compiled in-graph
    engine, where the trace oracle is unavailable. Two batches with
    identical prompts, params, and scheduling geometry are compared:
    one mixed (per-request specs and unsteered twins) and one fully
    unsteered. Same geometry means byte-comparable outputs, so each
    request is judged against its own unsteered counterpart — different
    lengths, different specs, one batch."""

    def test_each_request_applies_only_its_own_spec(self, llm):
        prompts = [
            list(range(200, 248)),   # 0: unsteered twin, 48 tokens
            list(range(500, 548)),   # 1: late generation window
            list(range(600, 648)),   # 2: head position
            list(range(400, 436)),   # 3: unsteered twin, 36 tokens
            list(range(700, 736)),   # 4: tail position on short prompt
        ]
        sp = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        tp = [TokensPrompt(prompt_token_ids=p) for p in prompts]
        steering = [
            None,
            spec(phases=["generation"], generation_window=(2, 4)),
            spec(phases=["prompt"], positions=[0, 1, 2, 3]),
            None,
            spec(phases=["prompt"], positions=[-1]),
        ]
        llm.reset_prefix_cache()
        mixed = llm.generate(tp, [sp] * 5, steering=steering, use_tqdm=False)
        llm.reset_prefix_cache()
        plain = llm.generate(tp, [sp] * 5, use_tqdm=False)
        m = [list(o.outputs[0].token_ids) for o in mixed]
        p = [list(o.outputs[0].token_ids) for o in plain]

        assert m[0] == p[0] and m[3] == p[3], (
            "unsteered requests were contaminated by co-batched steering"
        )
        assert m[1][:2] == p[1][:2], (
            "(2,4)-window request diverged before its window opened"
        )
        assert m[1] != p[1], "window spec produced no effect in the batch"
        assert m[2] != p[2], "head-range spec produced no effect"
        assert m[4] != p[4], (
            "positions=[-1] did not resolve against the shorter "
            "request's own prompt length"
        )
