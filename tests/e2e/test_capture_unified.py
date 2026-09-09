# SPDX-License-Identifier: Apache-2.0
"""Unified engine: capture on a compiled, prefix-caching, steering engine.

One engine serves steering and capture. While capture is idle the
engine keeps compiled execution with full CUDA graphs (steering in
full-graph mode); eligible capture batches use a separate FULL graph,
with raw eager execution for other batches needing rows. Admission skips
prefix reads only when a hit could omit selected prompt rows; cache
writes retain their ordinary keys for reuse after capture.
"""

import os

import torch
from vllm import SamplingParams

from helpers import DENSE_MODEL, steering_spec

ENGINE_PROFILE = "dense_graph_capture"

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_algorithms=["direct"],
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


def test_repeated_capture_has_complete_labels_and_clears_stream(llm):
    """Each helper call owns one complete result and stops its capture stream."""
    import easysteer.hidden_states as hs
    from vllm.capture import match_capture_request_id

    hidden_size = llm.llm_engine.vllm_config.model_config.hf_config.hidden_size
    previous_ids = set()
    for _ in range(2):
        result = hs.capture(
            llm, [PROMPT], layers=[10], max_tokens=1, ignore_eos=True,
        )
        assert result.labelled and len(result) == 1
        assert result.layer_ids == [10]
        output = result.outputs[0]
        prompt_ids = list(output.prompt_token_ids)
        assert len(output.outputs[0].token_ids) == 1
        assert result.sample_positions(0) == list(range(len(prompt_ids)))
        assert result.sample_token_ids(0) == prompt_ids
        rows = result.rows(10)
        assert tuple(rows.shape) == (len(prompt_ids), hidden_size)
        assert torch.isfinite(rows).all().item()
        request_ids = set(result.meta(10).req_ids)
        assert request_ids and request_ids.isdisjoint(previous_ids)
        assert all(match_capture_request_id(rid, output.request_id)
                   for rid in request_ids)
        previous_ids.update(request_ids)
        assert not rpc(llm, "capture_status", "hidden_states")[0]["enabled"]
        assert rpc(llm, "fetch_captured", "hidden_states", clear=False)[0] == {}


def test_warm_cache_capture_is_complete(llm):
    """Capture recomputes selected prompt rows even with a warm cache."""
    import easysteer.hidden_states as hs

    llm.generate(
        LONG_PROMPT, SamplingParams(max_tokens=1), use_tqdm=False
    )
    result = hs.capture(llm, [LONG_PROMPT], max_tokens=1, layers=[10])
    plen = len(result.outputs[0].prompt_token_ids)
    assert result.sample_positions(0) == list(range(plen))
    assert result.outputs[0].num_cached_tokens == 0


def test_rpc_capture_recomputes_and_preserves_cache_reuse(llm):
    from vllm.capture import deserialize_captured

    prompt = LONG_PROMPT + "Water is made of hydrogen and oxygen atoms. "
    rpc(llm, "start_capture", "hidden_states", layers=[10])
    try:
        out = llm.generate(prompt, SamplingParams(max_tokens=1), use_tqdm=False)[0]
        assert out.num_cached_tokens == 0
        raw = rpc(llm, "fetch_captured", "hidden_states", clear=True)[0]
    finally:
        rpc(llm, "stop_capture", "hidden_states")
    tensors, meta = deserialize_captured(raw)
    assert tensors[10].shape[0] == len(out.prompt_token_ids)
    assert meta[10].positions.tolist() == list(range(len(out.prompt_token_ids)))
    warm = llm.generate(prompt, SamplingParams(max_tokens=1), use_tqdm=False)[0]
    assert warm.num_cached_tokens > 0, "capture must populate reusable cache blocks"


def test_generation_only_capture_tolerates_cache_hits(llm):
    """Selections that cannot touch prompt rows are unaffected by hits."""
    from vllm.model_hooks.steering.api import SelectSpec

    prompt = LONG_PROMPT + "The tallest mountain on Earth is Everest. "
    llm.generate(prompt, SamplingParams(max_tokens=1), use_tqdm=False)
    rpc(
        llm,
        "start_capture",
        "hidden_states",
        layers=[10],
        select=SelectSpec(generation="all").to_wire(),
    )
    try:
        out = llm.generate(
            prompt, SamplingParams(max_tokens=4, ignore_eos=True), use_tqdm=False
        )[0]
        assert out.num_cached_tokens > 0
        raw = rpc(llm, "fetch_captured", "hidden_states", clear=True)[0]
    finally:
        rpc(llm, "stop_capture", "hidden_states")
    from vllm.capture import deserialize_captured

    tensors, meta = deserialize_captured(raw)
    assert tensors[10].shape[0] == 3, "the three decode forwards"


def test_helper_per_prompt_selection_controls_cache_reads(llm):
    import easysteer.hidden_states as hs
    from vllm.steer_vectors import SelectSpec

    prompts = [LONG_PROMPT + suffix for suffix in ("First.", "Second.", "Third.")]
    llm.generate(prompts, SamplingParams(max_tokens=1), use_tqdm=False)
    result = hs.capture(
        llm, prompts, layers=[10], max_tokens=3, ignore_eos=True,
        select=SelectSpec(generation="all"),
        per_prompt_selects=[
            SelectSpec(prompt="all"), SelectSpec(prompt_positions=[-1]), None
        ],
    )
    lengths = [len(out.prompt_token_ids) for out in result.outputs]
    assert result.outputs[0].num_cached_tokens == 0
    assert all(out.num_cached_tokens > 0 for out in result.outputs[1:])
    assert result.sample_positions(0) == list(range(lengths[0]))
    assert result.sample_positions(1) == [lengths[1] - 1]
    assert result.sample_positions(2) == [lengths[2], lengths[2] + 1]


def test_capture_replays_graph_after_helper_stop_and_restart(llm):
    import easysteer.hidden_states as hs
    from vllm.steer_vectors import SelectSpec

    before = rpc(llm, "capture_status", "hidden_states")[0]
    for _ in range(2):
        result = hs.capture(
            llm, [LONG_PROMPT], layers=[10], max_tokens=4, ignore_eos=True,
            select=SelectSpec(generation="all"),
        )
        plen = len(result.outputs[0].prompt_token_ids)
        assert result.sample_positions(0) == list(range(plen, plen + 3))
    after = rpc(llm, "capture_status", "hidden_states")[0]
    assert not after["enabled"] and after["graph_ready"]
    assert after["graph_replays"] >= before["graph_replays"] + 6
    assert after["graph_buffer_bytes"] > 0


def test_capture_and_steering_coexist(llm):
    """Steering applies on capture-dispatched batches; rows are captured."""
    from vllm.capture import deserialize_captured

    spec = steering_spec(scale=2.0, layers=list(range(10, 26)))
    sp = SamplingParams(temperature=0.0, max_tokens=24)
    plain = llm.generate(PROMPT, sp, use_tqdm=False)[0].outputs[0].text
    rpc(llm, "start_capture", "hidden_states", layers=[12])
    try:
        steered = llm.generate(
            PROMPT,
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


def test_capture_budget_keeps_ordinary_graphs_until_request_drain(llm):
    """Budget drops count selected rows per layer, including ordinary replays."""
    from vllm.capture import deserialize_captured

    rpc(
        llm, "start_capture", "hidden_states", layers=[10],
        select={"generation": "all"}, budget_rows=2,
    )
    try:
        first = llm.generate(
            PROMPT, SamplingParams(max_tokens=4, ignore_eos=True), use_tqdm=False
        )[0]
        full = rpc(llm, "capture_status", "hidden_states")[0]
        assert full["tokens_stored"] == 2 and full["tokens_dropped"] == 1
        assert full["graph_ready"]
        llm.generate(
            PROMPT, SamplingParams(max_tokens=3, ignore_eos=True), use_tqdm=False
        )
        idle = rpc(llm, "capture_status", "hidden_states")[0]
        assert idle["tokens_stored"] == 2 and idle["tokens_dropped"] == 3
        assert idle["graph_replays"] == full["graph_replays"]
        assert idle["eager_capture_forwards"] == full["eager_capture_forwards"]
        rpc(llm, "fetch_captured", "hidden_states", req_ids=[first.request_id])
        resumed = llm.generate(
            PROMPT, SamplingParams(max_tokens=3, ignore_eos=True), use_tqdm=False
        )[0]
        active = rpc(llm, "capture_status", "hidden_states")[0]
        assert active["graph_replays"] == idle["graph_replays"] + 2
        assert active["tokens_stored"] == 2 and active["tokens_dropped"] == 3
        raw = rpc(llm, "fetch_captured", "hidden_states")[0]
        tensors, meta = deserialize_captured(raw)
        assert tensors[10].shape[0] == 2
        plen = len(resumed.prompt_token_ids)
        assert meta[10].positions.tolist() == [plen, plen + 1]
    finally:
        rpc(llm, "stop_capture", "hidden_states")
