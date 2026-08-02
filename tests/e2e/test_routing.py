# SPDX-License-Identifier: Apache-2.0
"""Per-request slot routing: multi-vector configs and batched isolation.

One eager dense engine, greedy sampling. Covers:
- a one-entry multi-vector spec is byte-identical to the equivalent
  single-vector spec (same math through the multi path);
- a two-vector spec (split layers, sequential) steers;
- no cross-request contamination in a [steered, plain] batch;
- a 51-scale sweep: batch-repeat determinism and scale-0 == unsteered.

Batched-vs-sequential byte equality is deliberately NOT asserted: vLLM
batch-shape numerics make it approximate (a long-standing, confirmed
expectation). The gates are repeat determinism (same batch twice must
match exactly) and the anchors; a real routing bug shows up there.

async_scheduling is pinned off: byte-comparing one request's output
across generate calls needs identical batch geometry, and async
admission makes prefill co-batching timing-dependent.
"""

import pytest

from vllm import SamplingParams

from helpers import DENSE_MODEL, DENSE_VECTOR, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    enforce_eager=True,
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.3,
    max_model_len=2048,
    max_num_seqs=64,
    async_scheduling=False,
)

PROMPT = (
    "<|im_start|>user\nAlice's dog has passed away. "
    "Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
)
LAYERS = list(range(10, 26))


def gen(llm, prompts, steering, max_tokens=96):
    # ignore_eos keeps batch geometry identical across compared runs
    # (early EOS changes co-batching -> numeric drift, not a real diff).
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    outs = llm.generate(prompts, steering=steering, sampling_params=sp,
                        use_tqdm=False)
    return [o.outputs[0].text for o in outs]


@pytest.fixture(scope="module")
def plain(llm):
    return gen(llm, [PROMPT], None)[0]


@pytest.fixture(scope="module")
def two_vector_spec():
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    both = ApplySpec(phases=["prompt", "generation"])
    return SteeringSpec(
        conflict="sequential",
        vectors=[
            VectorSpec(source=DENSE_VECTOR, scale=1.2,
                       layers=list(range(10, 18)), apply=both),
            VectorSpec(source=DENSE_VECTOR, scale=1.2,
                       layers=list(range(18, 26)), apply=both),
        ],
    )


class TestMultiVector:
    def test_single_entry_multi_equals_single(self, llm):
        from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

        single = gen(llm, [PROMPT], steering_spec(scale=2.0, layers=LAYERS))[0]
        multi_one = gen(
            llm,
            [PROMPT],
            SteeringSpec(vectors=[VectorSpec(
                source=DENSE_VECTOR, scale=2.0, layers=LAYERS,
                apply=ApplySpec(phases=["prompt", "generation"]),
            )]),
        )[0]
        assert multi_one == single, (
            "one-entry multi-vector spec must equal the single-vector spec"
        )

    def test_two_vector_spec_steers(self, llm, plain, two_vector_spec):
        assert gen(llm, [PROMPT], two_vector_spec)[0] != plain

    def test_no_cross_request_contamination(self, llm, two_vector_spec):
        mixed = gen(llm, [PROMPT, PROMPT], [two_vector_spec, None])
        plain_batch = gen(llm, [PROMPT, PROMPT], None)
        assert mixed[1] == plain_batch[1], (
            "plain request contaminated by the co-batched multi-vector request"
        )


SCALES = [round(i * 0.1, 1) for i in range(51)]


def _scale_spec(scale):
    return steering_spec(scale=scale, layers=LAYERS)


@pytest.fixture(scope="module")
def sweep(llm):
    llm.preload_steer_vectors([DENSE_VECTOR])
    unsteered = gen(llm, [PROMPT], None, max_tokens=64)[0]
    ref = {
        s: gen(llm, [PROMPT], _scale_spec(s), max_tokens=64)[0]
        for s in SCALES
    }
    batch = dict(zip(SCALES, gen(
        llm, [PROMPT] * len(SCALES),
        [_scale_spec(s) for s in SCALES], max_tokens=64,
    )))
    batch2 = dict(zip(SCALES, gen(
        llm, [PROMPT] * len(SCALES),
        [_scale_spec(s) for s in SCALES], max_tokens=64,
    )))
    return unsteered, ref, batch, batch2


class TestScaleSweep:
    def test_batch_repeat_determinism(self, sweep):
        _, _, batch, batch2 = sweep
        mismatched = [s for s in SCALES if batch[s] != batch2[s]]
        assert not mismatched, f"batch repeat differed at scales {mismatched}"

    def test_scale_zero_is_unsteered_and_sweep_varies(self, sweep):
        unsteered, ref, _, _ = sweep
        assert ref[0.0] == unsteered, "scale 0.0 must equal the unsteered output"
        assert len(set(ref.values())) > 5, "sweep outputs barely vary"

    def test_batched_sweep_produces_scale_dependent_outputs(self, sweep):
        _, _, batch, _ = sweep
        assert len(set(batch.values())) > 5, (
            "batched sweep outputs barely vary across scales"
        )
