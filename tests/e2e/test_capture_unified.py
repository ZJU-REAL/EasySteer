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
from contextlib import contextmanager

import pytest
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
    worker_extension_cls="helpers.CaptureGraphWorkerExtension",
)
if graph_mode := os.environ.get("STEER_TEST_CAPTURE_GRAPH_MODE"):
    assert graph_mode in ("FULL", "FULL_DECODE_ONLY")
    ENGINE_KWARGS["compilation_config"] = {"cudagraph_mode": graph_mode}
if attention_backend := os.environ.get("STEER_TEST_ATTENTION_BACKEND"):
    ENGINE_KWARGS["attention_config"] = {"backend": attention_backend}

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


@contextmanager
def eager_capture(llm):
    """Temporarily select raw eager dispatch through the named test extension."""
    try:
        assert all(rpc(llm, "capture_test_set_eager", True))
        yield
    finally:
        assert all(rpc(llm, "capture_test_set_eager", False))


def assert_graph_replays(before, after, minimum=1):
    old = {status["topology"]["tp_rank"]: status for status in before}
    deltas = []
    for status in after:
        rank = status["topology"]["tp_rank"]
        assert status["graph_ready"]
        delta = status["graph_replays"] - old[rank]["graph_replays"]
        assert delta >= minimum
        deltas.append(delta)
    assert len(set(deltas)) == 1, "all TP ranks must replay the same capture steps"


def test_engine_keeps_full_cudagraphs(llm):
    cc = llm.llm_engine.vllm_config.compilation_config
    assert cc.cudagraph_mode.has_full_cudagraphs(), (
        "capture support must not cost the idle engine its full CUDA "
        "graphs"
    )
    assert rpc(llm, "capture_status", "hidden_states")[0]["hooked_layers"] == 0


def test_capture_on_compiled_engine(llm):
    import easysteer.capture as hs

    result = hs.capture(
        llm, ["The capital of France is", PROMPT], max_tokens=4, ignore_eos=True
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
    import easysteer.capture as hs
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
        statuses = rpc(llm, "capture_status", "hidden_states")
        assert all(not status["enabled"] and status["hooked_layers"] == 0
                   for status in statuses)
        assert all(raw == {} for raw in rpc(
            llm, "fetch_captured", "hidden_states", clear=False
        ))


@pytest.mark.parametrize("stream", ["hidden_states", "attention_heads"])
def test_public_capture_matches_worker_ownership_and_global_layout(llm, stream):
    """Graph results match eager shards and labels with owner-only replicas."""
    from easysteer.capture import capture
    from vllm.capture import deserialize_captured
    from vllm.inputs import TokensPrompt

    cfg = llm.llm_engine.vllm_config
    tp_size = cfg.parallel_config.tensor_parallel_size
    hf_config = cfg.model_config.hf_config
    token = llm.get_tokenizer().encode(" the", add_special_tokens=False)[0]
    options = dict(max_tokens=3, temperature=0.0, ignore_eos=True,
                   allowed_token_ids=[token])
    # An exact graph bucket keeps prefill geometry equal in FULL and eager mode.
    prompt = TokensPrompt(prompt_token_ids=list(range(400, 408)))
    spec = steering_spec(scale=0.25, layers=[10])
    before = rpc(llm, "capture_status", stream)
    result = capture(
        llm, [prompt], stream=stream, layers=[10], steering=spec, **options
    )
    graph_statuses = rpc(llm, "capture_status", stream)
    assert_graph_replays(before, graph_statuses, minimum=2)
    with eager_capture(llm):
        rpc(llm, "start_capture", stream, layers=[10])
        try:
            output = llm.generate(
                prompt, SamplingParams(**options), steering=spec, use_tqdm=False
            )[0]
            statuses = rpc(llm, "capture_status", stream)
            worker_rows = rpc(llm, "fetch_captured", stream)
        finally:
            rpc(llm, "stop_capture", stream)
    graph_by_rank = {status["topology"]["tp_rank"]: status for status in graph_statuses}
    for eager_status in statuses:
        rank = eager_status["topology"]["tp_rank"]
        graph_status = graph_by_rank[rank]
        assert eager_status["graph_replays"] == graph_status["graph_replays"]
        owns_rows = stream == "attention_heads" or rank == 0
        if owns_rows:
            assert (
                eager_status["eager_capture_forwards"]
                > graph_status["eager_capture_forwards"]
            )
        assert (graph_status["graph_buffer_bytes"] > 0) == owns_rows

    width = hf_config.hidden_size
    if stream == "attention_heads":
        heads = hf_config.num_attention_heads
        head_size = getattr(hf_config, "head_dim", None) or width // heads
        width = heads * head_size
        assert result.layouts[10] == dict(
            width=width, num_heads=heads, head_size=head_size
        )
    expected_tokens = list(output.prompt_token_ids)
    expected_tokens += list(output.outputs[0].token_ids)[:-1]
    assert result.sample_token_ids(0) == expected_tokens
    assert result.sample_positions(0) == list(range(len(expected_tokens)))
    global_rows = result.sample_rows(0, 10)
    assert global_rows.shape == (len(expected_tokens), width)
    exporters = [raw for raw in worker_rows if raw]
    expected_exporters = tp_size if stream == "attention_heads" else 1
    assert len(exporters) == expected_exporters
    if stream == "hidden_states":
        for status in statuses:
            rank = status["topology"]["tp_rank"]
            assert status["tokens_stored"] == (len(expected_tokens) if rank == 0 else 0)
    ranks = set()
    for raw in exporters:
        rows, meta = deserialize_captured(raw)
        rank = raw[10].get("shard", {}).get("tp_rank", 0)
        ranks.add(rank)
        local_width = width // expected_exporters
        start = rank * local_width
        assert rows[10].shape == (len(expected_tokens), local_width)
        assert meta[10].positions.tolist() == result.sample_positions(0)
        assert meta[10].token_ids.tolist() == expected_tokens
        if tp_size > 1:
            assert raw[10]["shard"]["feature_start"] == start
        tolerance = 2 * torch.finfo(rows[10].dtype).eps
        torch.testing.assert_close(
            global_rows[:, start:start + local_width], rows[10],
            rtol=tolerance, atol=tolerance,
        )
    assert ranks == set(range(expected_exporters))


@pytest.mark.parametrize("stream", ["hidden_states", "attention_heads"])
def test_full_capture_graph_replays_prefill_and_decode(llm, stream):
    """Exact bucket shapes isolate FULL prefill replay from decode replay."""
    from easysteer.capture import capture
    from vllm.inputs import TokensPrompt

    modes = {state["idle_mode"] for state in rpc(llm, "capture_test_graph_state")}
    assert len(modes) == 1
    if modes != {"FULL"}:
        pytest.skip("attention backend did not resolve to FULL prefill graphs")
    prompt_ids = list(range(400, 408))
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    options = dict(
        stream=stream, layers=[10], temperature=0, ignore_eos=True,
        allowed_token_ids=[400], steering=steering_spec(scale=0.25, layers=[10]),
    )
    before = rpc(llm, "capture_status", stream)
    prefill = capture(llm, [prompt], max_tokens=1, **options)
    after_prefill = rpc(llm, "capture_status", stream)
    assert_graph_replays(before, after_prefill)
    previous = {status["topology"]["tp_rank"]: status for status in before}
    for status in after_prefill:
        assert status["graph_replays"] == (
            previous[status["topology"]["tp_rank"]]["graph_replays"] + 1
        )
    assert prefill.sample_positions(0) == list(range(8))
    assert prefill.sample_token_ids(0) == prompt_ids

    graph = capture(llm, [prompt], max_tokens=2, **options)
    after_decode = rpc(llm, "capture_status", stream)
    assert_graph_replays(after_prefill, after_decode, minimum=2)
    with eager_capture(llm):
        eager = capture(llm, [prompt], max_tokens=2, **options)
    assert graph.sample_positions(0) == eager.sample_positions(0) == list(range(9))
    assert graph.sample_token_ids(0) == eager.sample_token_ids(0) == prompt_ids + [400]
    tolerance = 2 * torch.finfo(graph.rows(10).dtype).eps
    torch.testing.assert_close(
        graph.rows(10), eager.rows(10), rtol=tolerance, atol=tolerance,
    )


@pytest.mark.parametrize("stream", ["hidden_states", "attention_heads"])
def test_capture_byte_budget_failure_allows_next_capture(llm, stream):
    """A failed fetch must drain worker replies and leave the engine usable."""
    from easysteer.capture import capture

    options = dict(stream=stream, layers=[10], max_tokens=3, ignore_eos=True)
    before = rpc(llm, "capture_status", stream)
    # The engine utility RPC transports worker errors as a plain Exception.
    with pytest.raises(Exception, match="budget|discard|incomplete"):
        capture(llm, [PROMPT], budget_bytes=1, **options)
    after = rpc(llm, "capture_status", stream)
    assert all(not status["enabled"] for status in after)
    if llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size > 1:
        assert_graph_replays(before, after, minimum=2)
    result = capture(llm, [PROMPT], **options)
    plen = len(result.outputs[0].prompt_token_ids)
    assert result.sample_positions(0) == list(range(plen + 2))
    assert all(raw == {} for raw in rpc(llm, "fetch_captured", stream))


def test_warm_cache_capture_is_complete(llm):
    """Capture recomputes selected prompt rows even with a warm cache."""
    import easysteer.capture as hs

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
    import easysteer.capture as hs
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


@pytest.mark.parametrize("stream", ["hidden_states", "attention_heads"])
def test_capture_dispatch_after_helper_stop_and_restart(llm, stream):
    """Changing row selection reuses buffers and ignores unselected requests."""
    import easysteer.capture as hs
    from vllm.steer_vectors import SelectSpec

    before = rpc(llm, "capture_status", stream)
    graph_states = None
    for position in (0, 1):
        result = hs.capture(
            llm, [LONG_PROMPT], stream=stream, layers=[10],
            max_tokens=3, ignore_eos=True,
            select=SelectSpec(generation_positions=[position]),
        )
        plen = len(result.outputs[0].prompt_token_ids)
        assert result.sample_positions(0) == [plen + position]
        after = rpc(llm, "capture_status", stream)
        assert_graph_replays(before, after)
        assert all(not status["enabled"] and status["hooked_layers"] == 0
                   for status in after)
        current = rpc(llm, "capture_test_graph_state")
        assert all(state["mode"] in ("FULL", "FULL_DECODE_ONLY") for state in current)
        if graph_states is not None:
            assert current == graph_states, "selection changes must keep fixed graph buffers"
        graph_states, before = current, after

    # Staggered admission can combine selected decode with an unselected
    # prefill. That batch may correctly use eager capture in decode-only mode.
    mixed = hs.capture(
        llm, [LONG_PROMPT, PROMPT], stream=stream, layers=[10],
        max_tokens=3, ignore_eos=True,
        per_prompt_selects=[SelectSpec(generation_positions=[1]),
                            SelectSpec(generation_positions=[99])],
    )
    assert mixed.sample_positions(0) == [len(mixed.outputs[0].prompt_token_ids) + 1]
    assert mixed.sample_positions(1) == []
    assert mixed.sample_rows(1, 10).shape[0] == 0
    previous = {status["topology"]["tp_rank"]: status for status in before}
    replay_deltas = []
    for status in rpc(llm, "capture_status", stream):
        rank = status["topology"]["tp_rank"]
        old = previous[rank]
        delta = status["graph_replays"] - old["graph_replays"]
        replay_deltas.append(delta)
        if delta == 0 and (stream == "attention_heads" or rank == 0):
            assert status["eager_capture_forwards"] > old["eager_capture_forwards"]
    assert len(set(replay_deltas)) == 1


def test_capture_graph_replaces_layers_and_streams(llm):
    """A new component signature cannot replay stale layer or stream buffers."""
    from easysteer.capture import capture
    from vllm.capture import SelectSpec

    for stream, layer in (("hidden_states", 10), ("hidden_states", 11),
                          ("attention_heads", 11)):
        before = rpc(llm, "capture_status", stream)
        result = capture(
            llm, [PROMPT], stream=stream, layers=[layer], max_tokens=2,
            ignore_eos=True, select=SelectSpec(generation="all"),
        )
        assert result.layer_ids == [layer]
        assert result.sample_positions(0) == [len(result.outputs[0].prompt_token_ids)]
        assert_graph_replays(before, rpc(llm, "capture_status", stream))
        for state in rpc(llm, "capture_test_graph_state"):
            assert state["signature"] == [[stream, [layer]]]
            expected = {f"{stream}:{layer}"} if (
                stream == "attention_heads" or state["rank"] == 0
            ) else set()
            assert set(state["buffers"]) == expected


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


@pytest.mark.parametrize("stream", ["hidden_states", "attention_heads"])
def test_capture_row_budget_resumes_after_request_drain(llm, stream):
    """Budget drops count selected rows per layer, including ordinary replays."""
    from vllm.capture import assemble_captured

    tp_size = llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size
    rpc(
        llm, "start_capture", stream, layers=[10],
        select={"generation": "all"}, budget_rows=2,
    )
    try:
        first = llm.generate(
            PROMPT, SamplingParams(max_tokens=4, ignore_eos=True), use_tqdm=False
        )[0]
        full = rpc(llm, "capture_status", stream)
        for status in full:
            owner = stream == "attention_heads" or status["topology"]["tp_rank"] == 0
            assert status["tokens_stored"] == (2 if owner else 0)
            assert status["tokens_dropped"] == (1 if owner else 0)
            assert status["graph_ready"]
        llm.generate(
            PROMPT, SamplingParams(max_tokens=3, ignore_eos=True), use_tqdm=False
        )
        idle = rpc(llm, "capture_status", stream)
        full_by_rank = {status["topology"]["tp_rank"]: status for status in full}
        for status in idle:
            rank = status["topology"]["tp_rank"]
            old = full_by_rank[rank]
            owner = stream == "attention_heads" or rank == 0
            assert status["tokens_stored"] == (2 if owner else 0)
            assert status["tokens_dropped"] == (3 if owner else 0)
            assert status["graph_replays"] == (
                old["graph_replays"] + (2 if tp_size > 1 else 0)
            )
            assert status["eager_capture_forwards"] == old["eager_capture_forwards"]
        rpc(llm, "fetch_captured", stream, req_ids=[first.request_id])
        resumed = llm.generate(
            PROMPT, SamplingParams(max_tokens=3, ignore_eos=True), use_tqdm=False
        )[0]
        active = rpc(llm, "capture_status", stream)
        idle_by_rank = {status["topology"]["tp_rank"]: status for status in idle}
        for status in active:
            rank = status["topology"]["tp_rank"]
            old = idle_by_rank[rank]
            owner = stream == "attention_heads" or rank == 0
            assert status["graph_replays"] == old["graph_replays"] + 2
            assert status["tokens_stored"] == (2 if owner else 0)
            assert status["tokens_dropped"] == (3 if owner else 0)
        raw = rpc(llm, "fetch_captured", stream)
        tensors, meta, _ = assemble_captured(raw, tp_size=tp_size)
        assert tensors[10].shape[0] == 2
        plen = len(resumed.prompt_token_ids)
        assert meta[10].positions.tolist() == [plen, plen + 1]
    finally:
        rpc(llm, "stop_capture", stream)
