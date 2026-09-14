# SPDX-License-Identifier: Apache-2.0
"""In-graph steering at large slot capacity (per-token gather kernel).

Above ``max_steer_vectors > 2 * steer_graph_max_rank`` the low-rank
family switches from dense all-slot coefficients to per-token weight
gathers (the dense [slots, tokens, hidden] product grows linearly with
capacity and OOMs Inductor autotuning at a few hundred slots). This
module boots the same engine as test_fullgraph.py at capacity 96 so
every behavioral check runs through the gather formulation: low-rank
steering effects, zero-scale controls and co-batched plain requests.
"""

import os

from vllm import SamplingParams

from helpers import DENSE_MODEL, graph_replay

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_algorithms=["direct", "lm_steer", "loreft"],
    steer_graph_mode="in_graph",
    max_steer_vectors=96,  # > 2 * steer_graph_max_rank -> gather path
    steer_graph_max_rank=32,
    enforce_eager=False,
    tensor_parallel_size=int(os.environ.get("STEER_TEST_TP", "1")),
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.25,
    max_model_len=2048,
    max_num_seqs=8,
    async_scheduling=False,
    worker_extension_cls="helpers.CaptureGraphWorkerExtension",
)

TEXT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
LAYERS = list(range(10, 26))
SP = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
HIDDEN = 1536  # Qwen2.5-1.5B

LOREFT_WEIGHT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "replications", "loreft", "weight",
)


def _data_spec(payload, algorithm, scale, layers=None, **apply_kwargs):
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    if not apply_kwargs:
        apply_kwargs = {"prompt": "all", "generation": "all"}
    return SteeringSpec(vectors=[VectorSpec(
        data=payload, algorithm=algorithm, scale=scale, layers=layers,
        apply=ApplySpec(**apply_kwargs),
    )])


def gen(llm, prompts, **kwargs):
    outs = llm.generate(prompts, sampling_params=SP, use_tqdm=False, **kwargs)
    return [o.outputs[0].text for o in outs]


def test_lowrank_gather_path_steers_and_zero_is_exact(llm):
    """Compare a rank-4 lm_steer projector through the gather kernel
    with the unsteered and zero-scale controls."""
    import numpy as np

    from vllm.model_hooks.steering.payloads import LowRankProjector

    plain = gen(llm, [TEXT])[0]
    axes = np.zeros((HIDDEN, 4), dtype=np.float32)
    axes[:4, :4] = np.eye(4, dtype=np.float32)
    proj = LowRankProjector(axes, axes)
    with graph_replay(llm, "full"):
        steered = gen(llm, [TEXT],
                      steering=_data_spec(proj, "lm_steer", 50.0, LAYERS))[0]
    assert steered != plain, "lm_steer did not steer via the gather path"
    zero = gen(llm, [TEXT],
               steering=_data_spec(proj, "lm_steer", 0.0, LAYERS))[0]
    assert zero == plain, "zero-scale lm_steer is not a no-op (gather path)"


def test_loreft_decode_gather_path(llm):
    """LoReFT changes fixed decode rows and replays the ordinary gather graph."""
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
    with graph_replay(llm, "full", minimum=3):
        output = llm.generate(
            prompt, sampling_params=SamplingParams(**options), steering=spec,
            use_tqdm=False,
        )[0]
    assert list(output.outputs[0].token_ids) == [token] * 4


def test_many_distinct_configs_isolated(llm):
    """Distinct zero-scale gather configs preserve every fixed-input row."""
    import numpy as np
    import torch
    from vllm.inputs import TokensPrompt
    from vllm.model_hooks.steering.payloads import LowRankProjector

    from easysteer.hidden_states import capture

    config = llm.llm_engine.vllm_config.steer_vector_config
    assert config.max_steer_vectors > 2 * config.graph_max_rank
    axes = np.zeros((HIDDEN, 4), dtype=np.float32)
    axes[:4] = np.eye(4, dtype=np.float32)
    projector = LowRankProjector(axes, axes)
    prompt_ids = list(range(400, 408))
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids) for _ in range(8)]
    # A single prefill forward has no generated-input feedback or early EOS.
    options = dict(layers=[10], max_tokens=1, ignore_eos=True, temperature=0)
    plain = capture(llm, prompts, steering=False, **options)
    steering = [
        _data_spec(projector, "lm_steer", 0.0, [10], prompt_positions=[i])
        for i in range(7)
    ] + [False]
    mixed = capture(llm, prompts, steering=steering, **options)
    for index in range(8):
        assert mixed.sample_positions(index) == plain.sample_positions(index)
        assert mixed.sample_positions(index) == list(range(8))
        assert mixed.sample_token_ids(index) == plain.sample_token_ids(index)
        assert mixed.sample_token_ids(index) == prompt_ids
        assert torch.equal(
            mixed.sample_rows(index, 10), plain.sample_rows(index, 10)
        ), (
            f"zero-scale gather steering changed sample {index}"
        )
