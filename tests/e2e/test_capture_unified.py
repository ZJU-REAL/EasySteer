# SPDX-License-Identifier: Apache-2.0
"""Unified engine: capture on a compiled, prefix-caching, steering engine.

One engine serves steering and capture. While capture is idle the
engine keeps compiled execution with full CUDA graphs (steering in
full-graph mode); capture-active batches dispatch to the raw eager
forward where the capture hooks run natively. Prefix caching stays
enabled: capture requests carry a unique cache_salt (full recompute,
no cache hits), and unsalted requests that do hit the cache while
capture is enabled fail explicitly at fetch.
"""

import os

import pytest
import torch
from vllm import SamplingParams

from helpers import DENSE_MODEL, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    # Compiled engine; steering auto graph mode resolves to full, so
    # full CUDA graphs are kept while capture is idle.
    enforce_eager=False,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_prefix_caching=True,
    gpu_memory_utilization=0.18,
    max_model_len=2048,
)

PROMPT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
# Long enough that a repeat spans several 16-token cache blocks.
LONG_PROMPT = (
    "The quick brown fox jumps over the lazy dog near the riverbank "
    "while the morning sun rises slowly over the distant mountains and "
    "birds sing in the tall green trees of the quiet valley. "
)


def rpc(llm, method, *args, **kwargs):
    return llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)


def test_engine_keeps_full_cudagraphs(llm):
    cc = llm.llm_engine.vllm_config.compilation_config
    assert cc.cudagraph_mode.has_full_cudagraphs(), (
        "capture support must not cost the idle engine its full CUDA "
        "graphs"
    )


def test_capture_on_compiled_engine(llm):
    import easysteer.hidden_states as hs

    result = hs.capture(
        llm, ["The capital of France is", PROMPT], max_tokens=4
    )
    assert result.labelled
    assert len(result.layer_ids) > 20, "all decoder layers hooked"
    for i in range(2):
        plen = len(result.outputs[i].prompt_token_ids)
        positions = result.sample_positions(i)
        # Full prompt + the generated tokens that ran a forward
        # (max_tokens=4 -> 3 decode forwards).
        assert positions == list(range(plen + 3))


def test_capture_is_deterministic_across_calls(llm):
    import easysteer.hidden_states as hs

    r1 = hs.capture(llm, [PROMPT], layers=[10])
    r2 = hs.capture(llm, [PROMPT], layers=[10])
    assert torch.equal(r1.rows(10), r2.rows(10))


def test_warm_cache_capture_is_complete(llm):
    """A salted capture request never hits the cache it just warmed."""
    import easysteer.hidden_states as hs

    llm.generate(
        LONG_PROMPT, SamplingParams(max_tokens=1), use_tqdm=False
    )
    result = hs.capture(llm, [LONG_PROMPT], max_tokens=1, layers=[10])
    plen = len(result.outputs[0].prompt_token_ids)
    assert result.sample_positions(0) == list(range(plen))


def test_unsalted_cache_hit_fails_explicitly(llm):
    prompt = LONG_PROMPT + "Water is made of hydrogen and oxygen atoms. "
    llm.generate(prompt, SamplingParams(max_tokens=1), use_tqdm=False)
    rpc(llm, "start_capture", "hidden_states", layers=[10])
    try:
        llm.generate(prompt, SamplingParams(max_tokens=1), use_tqdm=False)
        with pytest.raises(Exception, match="cache_salt"):
            rpc(llm, "fetch_captured", "hidden_states", clear=True)
    finally:
        rpc(llm, "stop_capture", "hidden_states")


def test_generation_only_capture_tolerates_cache_hits(llm):
    """Selections that cannot touch prompt rows are unaffected by hits."""
    from vllm.steer_vectors.api import SelectSpec

    prompt = LONG_PROMPT + "The tallest mountain on Earth is Everest. "
    llm.generate(prompt, SamplingParams(max_tokens=1), use_tqdm=False)
    rpc(
        llm,
        "start_capture",
        "hidden_states",
        layers=[10],
        select=SelectSpec(phases=["generation"]).to_wire(),
    )
    try:
        llm.generate(prompt, SamplingParams(max_tokens=4), use_tqdm=False)
        raw = rpc(llm, "fetch_captured", "hidden_states", clear=True)[0]
    finally:
        rpc(llm, "stop_capture", "hidden_states")
    from vllm.capture import deserialize_captured

    tensors, meta = deserialize_captured(raw)
    assert tensors[10].shape[0] == 3, "the three decode forwards"


def test_capture_and_steering_coexist(llm):
    """Steering applies on capture-dispatched batches; rows are captured."""
    from vllm.capture import deserialize_captured

    spec = steering_spec(scale=2.0, layers=list(range(10, 26)))
    sp = SamplingParams(temperature=0.0, max_tokens=24)
    plain = llm.generate(PROMPT, sp, use_tqdm=False)[0].outputs[0].text
    rpc(llm, "start_capture", "hidden_states", layers=[12])
    try:
        steered = llm.generate(
            {"prompt": PROMPT, "cache_salt": "unified-coexist"},
            sp,
            steering=spec,
            use_tqdm=False,
        )[0].outputs[0].text
        raw = rpc(llm, "fetch_captured", "hidden_states", clear=True)[0]
    finally:
        rpc(llm, "stop_capture", "hidden_states")
    assert steered != plain, "steering must apply on the capture path"
    tensors, meta = deserialize_captured(raw)
    assert 12 in tensors and tensors[12].shape[0] > 0
