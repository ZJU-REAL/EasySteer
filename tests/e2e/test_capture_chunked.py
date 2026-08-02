# SPDX-License-Identifier: Apache-2.0
"""Capture coverage and reductions under chunked prefill.

Covers: positions='all' captures every prompt token exactly once across
chunks (129-token prompt, 64-token budget -> chunks 64/64/1) plus one
row per decode forward; positions='last' stores one row per logical
step (final prompt chunk + each decode step), not one per chunk,
including the 1-token-prompt edge case.
"""

import pytest
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

# vllm.hidden_states is a backward-compatibility alias of vllm.capture;
# imported on purpose so the alias stays covered.
from vllm.hidden_states import deserialize_hidden_states

from helpers import DENSE_MODEL

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enforce_eager=True,
    enable_prefix_caching=False,
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


def capture_rows(llm, prompt_ids, positions):
    rpc(llm, "start_capture", "hidden_states", layers=[0], positions=positions)
    try:
        llm.generate(
            TokensPrompt(prompt_token_ids=list(prompt_ids)), SP, use_tqdm=False
        )
        hs = deserialize_hidden_states(rpc(llm, "fetch_captured", "hidden_states"))
    finally:
        rpc(llm, "stop_capture", "hidden_states")
    return hs[0].shape[0]


@pytest.mark.parametrize(
    "prompt_ids, positions, expected",
    [
        # 129 prompt tokens (each captured once across chunks) + 3 decode
        pytest.param(PROMPT_LONG, "all", 132, id="all-covers-chunks-and-decode"),
        # 1 final-chunk row + 3 decode rows: per logical step, not per chunk
        pytest.param(PROMPT_LONG, "last", 4, id="last-per-step-not-per-chunk"),
        # 1-token prompt: 1 prefill row + 3 decode rows, same semantics
        pytest.param([100], "last", 4, id="last-one-token-prompt"),
    ],
)
def test_capture_rows_under_chunked_prefill(llm, prompt_ids, positions, expected):
    rows = capture_rows(llm, prompt_ids, positions)
    assert rows == expected, (
        f"positions={positions!r} captured {rows} rows, want {expected}"
    )
