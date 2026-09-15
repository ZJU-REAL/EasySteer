# SPDX-License-Identifier: Apache-2.0
"""Capture coverage and reductions under chunked prefill.

Covers: reduce='all' captures every prompt token exactly once across
chunks (129-token prompt, 64-token budget -> chunks 64/64/1) plus one
row per decode forward; reduce='last' stores one row per logical
step (final prompt chunk + each decode step), not one per chunk,
including the 1-token-prompt edge case.
"""

import os

import pytest
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from vllm.capture import deserialize_captured

from helpers import DENSE_MODEL

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enforce_eager=True,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=64,
    max_num_seqs=4,
    max_model_len=512,
    gpu_memory_utilization=0.18,
)

PROMPT_LONG = list(range(100, 229))  # 129 tokens -> chunks 64/64/1
SP = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)


def rpc(llm, method, *args, **kwargs):
    return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)[0]


def capture_rows(llm, prompt_ids, reduce):
    rpc(llm, "start_capture", "hidden_states", layers=[0], reduce=reduce)
    try:
        llm.generate(
            TokensPrompt(prompt_token_ids=list(prompt_ids)), SP, use_tqdm=False
        )
        hs = deserialize_captured(rpc(llm, "fetch_captured", "hidden_states"))[0]
    finally:
        rpc(llm, "stop_capture", "hidden_states")
    return hs[0].shape[0]


@pytest.mark.parametrize(
    "prompt_ids, reduce, expected",
    [
        # 129 prompt tokens (each captured once across chunks) + 3 decode
        pytest.param(PROMPT_LONG, "all", 132, id="all-covers-chunks-and-decode"),
        # 1 final-chunk row + 3 decode rows: per logical step, not per chunk
        pytest.param(PROMPT_LONG, "last", 4, id="last-per-step-not-per-chunk"),
        # 1-token prompt: 1 prefill row + 3 decode rows, same semantics
        pytest.param([100], "last", 4, id="last-one-token-prompt"),
    ],
)
def test_capture_rows_under_chunked_prefill(llm, prompt_ids, reduce, expected):
    rows = capture_rows(llm, prompt_ids, reduce)
    assert rows == expected, (
        f"reduce={reduce!r} captured {rows} rows, want {expected}"
    )


@pytest.mark.parametrize("stream", ["hidden_states", "attention_heads"])
def test_public_capture_selects_chunk_boundaries_with_warm_prefix(llm, stream):
    """Warm cached blocks cannot omit selected rows across 64-token chunks."""
    from easysteer.capture import capture
    from vllm.steer_vectors import SelectSpec

    prompt = TokensPrompt(prompt_token_ids=list(PROMPT_LONG))
    warmup = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    llm.generate(prompt, warmup, use_tqdm=False)
    selected_prompt = [0, 63, 64, 127, 128]
    result = capture(
        llm, [prompt], stream=stream, layers=[0], max_tokens=4, ignore_eos=True,
        select=SelectSpec(prompt_positions=selected_prompt, generation="all"),
    )
    output = result.outputs[0]
    assert output.num_cached_tokens == 0
    assert result.sample_positions(0) == selected_prompt + [129, 130, 131]
    expected_tokens = [PROMPT_LONG[pos] for pos in selected_prompt]
    expected_tokens += list(output.outputs[0].token_ids)[:-1]
    assert result.sample_token_ids(0) == expected_tokens
    assert result.rows(0).shape[0] == len(expected_tokens)
    warm = llm.generate(prompt, warmup, use_tqdm=False)[0]
    assert warm.num_cached_tokens > 0, "capture must leave reusable prefix blocks"
