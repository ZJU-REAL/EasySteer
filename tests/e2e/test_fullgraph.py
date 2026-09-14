# SPDX-License-Identifier: Apache-2.0
"""Tier-1 in-graph steering (pinned; this workload declares
conditional algorithms, which auto resolves to split).

The steering kernel families (additive, projection, low-rank, replace)
read persistent buffers and capture into full CUDA graphs;
triggers/routing are computed host-side each step. Covers: full
cudagraphs kept (no piecewise downgrade); direct steering fires with
scale-0 no-op behavior and mixed-batch completion/effect; every non-direct
kernel family (erase, replace,
concept_replace, loreft — the replication checkpoint — and
lm_steer) steers under full graphs; normalized steering is effective and
over-rank payloads reject with an actionable error. Fixed-input kernel
tests cover exact replay and row isolation.

All steering uses v2 SteeringSpec; STEER_TEST_EAGER=1 runs the same
kernel path eagerly (skipping the cudagraph-mode check).
"""

import os
from contextlib import nullcontext

import pytest
from helpers import DENSE_MODEL, graph_replay, steering_spec
from vllm import SamplingParams

EAGER = os.environ.get("STEER_TEST_EAGER", "0") == "1"

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    # The module exercises every kernel family, including the
    # conditional ones (loreft, lm_steer) that auto resolves
    # pessimistically to split — so pin the in-graph tier explicitly
    # (the expert path; boots with the conditional-algorithms warning).
    steer_algorithms=[
        "attention_add", "concept_replace", "direct", "erase", "lm_steer", "loreft",
        "replace",
    ],
    steer_graph_mode="in_graph",
    # Pin the pre-default capacity: <= 2 * steer_graph_max_rank keeps
    # this module on the dense low-rank path (the gather path has its
    # own module, test_fullgraph_large_capacity.py).
    max_steer_vectors=8,
    enforce_eager=EAGER,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.25,
    max_model_len=2048,
    worker_extension_cls="helpers.CaptureGraphWorkerExtension",
)
if EAGER:
    # Eager debug path: in_graph on a non-compiled engine is normally
    # rejected at boot; this test-only flag exercises the same steering
    # kernels without CUDA-graph replay.
    os.environ["VLLM_STEER_EAGER_IN_GRAPH"] = "1"

TEXT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
LAYERS = list(range(10, 26))
# ignore_eos fixes the generation length for these comparisons.
SP = SamplingParams(temperature=0.0, max_tokens=96, ignore_eos=True)


def happy_spec():
    return steering_spec(scale=2.0, layers=LAYERS)


def gen(llm, prompts, **kwargs):
    outs = llm.generate(prompts, sampling_params=SP, use_tqdm=False, **kwargs)
    return [o.outputs[0].text for o in outs]


@pytest.fixture(scope="module")
def outs(llm):
    """Single-prompt and mixed-batch outputs on one engine."""
    spec = happy_spec()
    return {
        "plain": gen(llm, [TEXT])[0],
        "zero": gen(llm, [TEXT], steering=steering_spec(scale=0.0,
                                                        layers=LAYERS))[0],
        "happy": gen(llm, [TEXT], steering=spec)[0],
        # Mixed batch: one steered + one plain request in ONE batch
        # (None entries in the steering sequence leave prompts unsteered).
        "batch_mixed": gen(
            llm,
            [TEXT, TEXT],
            steering=[happy_spec(), None],
        ),
    }


@pytest.mark.skipif(EAGER, reason="STEER_TEST_EAGER=1 runs the kernel eagerly")
def test_full_cudagraphs_kept(llm):
    """in_graph mode must keep full CUDA graphs (no piecewise
    downgrade of vLLM's cudagraph_mode)."""
    cfg = llm.llm_engine.vllm_config
    assert cfg.steer_vector_config.graph_mode == "in_graph", (
        f"graph mode is {cfg.steer_vector_config.graph_mode!r}"
    )
    comp = cfg.compilation_config
    assert comp.cudagraph_mode.has_full_cudagraphs(), (
        f"cudagraph_mode {comp.cudagraph_mode} has no full graphs"
    )


def test_steering_fires(outs):
    assert outs["happy"] != outs["plain"], (
        "steered output identical to unsteered"
    )


def test_zero_scale_identical_to_no_steering(outs):
    assert outs["zero"] == outs["plain"], (
        "scale-0 steering differs from no steering"
    )


def test_mixed_batch_completes_and_steers(outs):
    """Mixed traffic completes with a visible nonzero steering effect.

    Exact row isolation and replay are checked on fixed-input kernels and
    slot state; this end-to-end smoke does not infer routing from text parity.
    """
    mixed = outs["batch_mixed"]
    assert len(mixed) == 2 and all(mixed)
    assert mixed[0] != mixed[1], "the mixed-batch steering effect is missing"


@pytest.mark.parametrize(
    "num_prompts,max_tokens", [(1, 4), (2, 1)],
    ids=["single_decode", "mixed_prefill"],
)
def test_attention_head_delta_is_isolated(llm, num_prompts, max_tokens):
    """Check token isolation in decode and request isolation in prefill.

    Mixed requests can enter separate decode batches across capture calls;
    their batch-dependent rounding is unsuitable for a direct delta oracle.
    """
    import numpy as np
    import torch
    from vllm.model_hooks.steering.payloads import DirectionVector
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    from easysteer.hidden_states import capture

    tp_size = llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size
    layer = 10
    token = llm.get_tokenizer().encode(" the", add_special_tokens=False)[0]
    options = dict(
        layers=[layer], stream="attention_heads", max_tokens=max_tokens,
        temperature=0.0, ignore_eos=True, allowed_token_ids=[token],
    )
    prompts = [TEXT] * num_prompts
    baseline = capture(llm, prompts, steering=False, **options)
    layout = baseline.layouts[layer]
    direction = np.zeros(layout["width"], dtype=np.float32)
    # Distinct directions on either side of the rank-0/rank-1 boundary
    # catch both rank-0-only application and reuse of the same local slice.
    boundary = layout["width"] // max(tp_size, 2)
    head_size = layout["head_size"]
    direction[boundary - head_size:boundary] = 0.25
    direction[boundary:boundary + head_size] = -0.5
    spec = SteeringSpec(vectors=[VectorSpec(
        data=DirectionVector({layer: direction}), algorithm="attention_add",
        scale=2.0, layers=[layer],
        apply=ApplySpec(prompt_positions=[-1], generation_positions=[0]),
    )])
    steering = [False] * (num_prompts - 1) + [spec]
    result = capture(llm, prompts, steering=steering, **options)
    for index in range(num_prompts):
        reference = baseline.sample(index)[layer]
        actual = result.sample(index)[layer]
        assert result.sample_positions(index) == baseline.sample_positions(index)
        assert result.sample_token_ids(index) == baseline.sample_token_ids(index)
        expected = reference.clone()
        selected = torch.zeros(reference.shape[0], dtype=torch.bool)
        if index == num_prompts - 1:
            prompt_len = len(result.outputs[index].prompt_token_ids)
            selected = torch.tensor([
                position in (prompt_len - 1, prompt_len)
                for position in result.sample_positions(index)
            ])
            expected[selected] += torch.from_numpy(direction).to(expected.dtype) * 2
            assert selected.sum() == (2 if max_tokens > 1 else 1)
            assert (actual[selected] - reference[selected]).abs().max() > 0.5
        tolerance = 2 * torch.finfo(actual.dtype).eps
        context = (
            f"attention_heads sample={index}, layer={layer}, TP={tp_size}, "
            f"selected_rows={selected.nonzero().flatten().tolist()}"
        )
        torch.testing.assert_close(
            actual, expected, rtol=tolerance, atol=tolerance,
            msg=lambda message: f"{context}\n{message}",
        )


def test_attention_steering_runs_in_ordinary_decode_graphs(llm, outs):
    """Generation-only head steering exercises measured ordinary graph replay."""
    import numpy as np
    from vllm.model_hooks.steering.payloads import DirectionVector

    config = llm.llm_engine.vllm_config.model_config.hf_text_config
    heads = config.num_attention_heads
    head_size = getattr(config, "head_dim", None) or config.hidden_size // heads
    direction = np.zeros(heads * head_size, dtype=np.float32)
    midpoint = (heads // 2) * head_size
    direction[midpoint - head_size:midpoint] = 0.25
    direction[midpoint:midpoint + head_size] = -0.5
    payload = DirectionVector({10: direction})
    zero = _data_spec(payload, "attention_add", 0.0, generation="all")
    assert gen(llm, [TEXT], steering=zero)[0] == outs["plain"]

    # Prompt processing stays unsteered; any difference must arise during decode.
    nonzero = _data_spec(payload, "attention_add", 16.0, generation="all")
    with nullcontext() if EAGER else graph_replay(llm, "full"):
        mixed = gen(llm, [TEXT, TEXT], steering=[nonzero, False])
    assert len(mixed) == 2 and all(mixed)
    assert mixed[0] != mixed[1], "attention steering had no effect during decode"


# ---------------------------------------------------------------------------
# Non-direct kernel families (projection, low-rank, replace)
# ---------------------------------------------------------------------------

LOREFT_WEIGHT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "replications", "loreft", "weight",
)
HIDDEN = 1536  # Qwen2.5-1.5B


def _data_spec(payload, algorithm, scale, layers=None, **apply_kwargs):
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    if not apply_kwargs:
        apply_kwargs = {"prompt": "all", "generation": "all"}
    return SteeringSpec(vectors=[VectorSpec(
        data=payload, algorithm=algorithm, scale=scale, layers=layers,
        apply=ApplySpec(**apply_kwargs),
    )])


def test_erase_steers_and_zero_payload_is_exact(llm, outs):
    """Compare erasing the happy direction with the unsteered and
    zero-scale controls."""
    erased = gen(llm, [TEXT], steering=steering_spec(
        algorithm="erase", scale=1.0, layers=LAYERS))[0]
    assert erased != outs["plain"], "erase did not change the output"
    zero = gen(llm, [TEXT], steering=steering_spec(
        algorithm="erase", scale=0.0, layers=LAYERS))[0]
    assert zero == outs["plain"], "zero-scale erase is not a no-op"


def test_replace_steers(llm, outs):
    """Replace family: substituting the hidden state with the vector on
    every selected row changes the output."""
    replaced = gen(llm, [TEXT], steering=steering_spec(
        algorithm="replace", scale=1.0, layers=[20]))[0]
    assert replaced != outs["plain"], "replace did not change the output"


def test_concept_replace_steers(llm, outs):
    """Projection family, concept_pair payload path."""
    import numpy as np
    from vllm.model_hooks.steering.payloads import ConceptPair, DirectionVector

    rng = np.random.RandomState(0)
    h1 = {la: rng.randn(HIDDEN).astype(np.float32) * 0.5 for la in LAYERS}
    h2 = {la: rng.randn(HIDDEN).astype(np.float32) * 50.0 for la in LAYERS}
    pair = ConceptPair(DirectionVector(h1), DirectionVector(h2))
    swapped = gen(llm, [TEXT],
                  steering=_data_spec(pair, "concept_replace", 1.0))[0]
    assert swapped != outs["plain"], "concept_replace did not change output"


def test_loreft_decode_effect_and_replay(llm):
    """The checkpoint changes fixed decode rows and runs in ordinary FULL graphs."""
    import torch

    from easysteer.hidden_states import capture
    from easysteer.vectors import from_pyreft

    prompt = "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n"
    token = llm.get_tokenizer().encode(" the", add_special_tokens=False)[0]
    options = dict(temperature=0.0, max_tokens=4, ignore_eos=True,
                   allowed_token_ids=[token])
    spec = _data_spec(from_pyreft(LOREFT_WEIGHT), "loreft", 1.0,
                      layers=[8], generation="all")
    plain = capture(llm, [prompt], layers=[8], steering=False, **options)
    steered = capture(llm, [prompt], layers=[8], steering=spec, **options)
    plen = len(plain.outputs[0].prompt_token_ids)
    assert steered.sample_positions(0) == plain.sample_positions(0)
    assert steered.sample_positions(0) == list(range(plen + 3))
    assert steered.sample_token_ids(0) == plain.sample_token_ids(0)
    before, after = plain.rows(8), steered.rows(8)
    assert torch.equal(before[:plen], after[:plen]), (
        "decode-only steering changed prefill"
    )
    assert not torch.allclose(before[plen:], after[plen:]), (
        "LoReFT decode rows are unchanged"
    )
    with nullcontext() if EAGER else graph_replay(llm, "full", minimum=3):
        output = llm.generate(
            prompt, sampling_params=SamplingParams(**options), steering=spec,
            use_tqdm=False,
        )[0]
    assert list(output.outputs[0].token_ids) == [token] * 4


def test_lm_steer_projection_and_zero_scale(llm, outs):
    """Compare a rank-4 lm_steer axis projector at high scale with the
    unsteered and zero-scale controls."""
    import numpy as np
    from vllm.model_hooks.steering.payloads import LowRankProjector

    axes = np.zeros((HIDDEN, 4), dtype=np.float32)
    axes[:4, :4] = np.eye(4, dtype=np.float32)
    proj = LowRankProjector(axes, axes)
    steered = gen(llm, [TEXT],
                  steering=_data_spec(proj, "lm_steer", 50.0, LAYERS))[0]
    assert steered != outs["plain"], "lm_steer did not change the output"
    zero = gen(llm, [TEXT],
               steering=_data_spec(proj, "lm_steer", 0.0, LAYERS))[0]
    assert zero == outs["plain"], "zero-scale lm_steer is not a no-op"


def test_normalized_steering_is_effective(llm, outs):
    """Normalized steering works; tensor tests verify the norm operation itself."""
    normed = gen(llm, [TEXT], steering=steering_spec(
        scale=2.0, layers=LAYERS, normalize=True))[0]
    assert normed and normed != outs["plain"], "normalized steering did not steer"


def test_over_rank_payload_rejected(llm):
    """Frontend rank rejection leaves the same engine usable."""
    import numpy as np
    from vllm.exceptions import VLLMClientError
    from vllm.model_hooks.steering.payloads import LowRankProjector

    limit = llm.llm_engine.vllm_config.steer_vector_config.graph_max_rank
    rank = limit + 1
    big = np.zeros((HIDDEN, rank), dtype=np.float32)
    with pytest.raises(
        VLLMClientError,
        match=f"payload rank {rank} above steer_graph_max_rank {limit}",
    ):
        gen(llm, [TEXT],
            steering=_data_spec(LowRankProjector(big, big), "lm_steer",
                                1.0, LAYERS))
    output = llm.generate(
        TEXT, sampling_params=SamplingParams(max_tokens=2, ignore_eos=True),
        steering=happy_spec(), use_tqdm=False,
    )[0].outputs[0]
    assert output.finish_reason == "length" and len(output.token_ids) == 2
