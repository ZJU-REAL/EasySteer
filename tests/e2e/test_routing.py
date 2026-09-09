# SPDX-License-Identifier: Apache-2.0
"""Per-request slot routing: multi-vector configs and batched isolation.

One eager dense engine, greedy sampling. Covers:
- a two-vector spec (split layers, sequential) steers;
- no cross-request contamination in a [steered, plain] batch;
- a representative scale sweep (51 scales with --steer-extended):
  scale-0 == unsteered, outputs vary across scales,
  and per-request isolation in the batched pass verified through the
  steering trace (every applied position belongs to a request routed to
  that apply's slot).

Isolation is checked through steering traces. async_scheduling is pinned
off to keep request scheduling consistent.
"""

import pytest
from helpers import DENSE_MODEL, DENSE_VECTOR, steering_spec
from vllm import SamplingParams

ENGINE_KWARGS = {
    "model": DENSE_MODEL,
    "enable_steer_vector": True,
    "steer_algorithms": ["direct"],
    "steer_multi_vector": True,
    "enforce_eager": True,
    "enable_chunked_prefill": False,
    "enable_prefix_caching": False,
    "gpu_memory_utilization": 0.3,
    "max_model_len": 2048,
    "max_num_seqs": 64,
    "async_scheduling": False,
}

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

    both = ApplySpec(prompt="all", generation="all")
    return SteeringSpec(
        conflict="sequential",
        vectors=[
            VectorSpec(source=DENSE_VECTOR, scale=1.2,
                       layers=list(range(10, 18)), apply=both),
            VectorSpec(source=DENSE_VECTOR, scale=1.2,
                       layers=list(range(18, 26)), apply=both),
        ],
    )


def test_pt_direction_payload_through_engine(llm, plain, tmp_path):
    """A .pt direction file steers through the client adapter + data=
    payload path (the engine no longer loads .pt files itself)."""
    import torch
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    import easysteer.vectors as vec

    pt = str(tmp_path / "direction.pt")
    torch.save(torch.randn(1536) * 0.02, pt)
    spec = SteeringSpec(vectors=[VectorSpec(
        data=vec.from_pt_direction(pt, layers=[10]),
        scale=8.0,
        apply=ApplySpec(prompt="all", generation="all"),
    )])
    out = gen(llm, [PROMPT], spec)[0]
    assert out != plain, "data= payload steering had no effect"


def test_wrong_source_format_rejected_at_admission(llm):
    """A non-GGUF source is rejected client-side with a clean error —
    it must never reach the worker (a worker-side load failure would
    kill the EngineCore)."""
    import pytest

    with pytest.raises(Exception, match="data="):
        gen(llm, [PROMPT], steering_spec(source="v.pt", scale=1.0, layers=[10]))


class TestMultiVector:
    def test_dynamic_conflict_finishes_only_affected_request(self, llm):
        """Generation-time overlap must not kill the engine or its batch peer."""
        from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

        spec = SteeringSpec(conflict="error", vectors=[
            VectorSpec(source=DENSE_VECTOR, layers=[10], scale=0.1,
                       apply=ApplySpec(generation_positions=[0]))
            for _ in range(2)
        ])
        outputs = llm.generate(
            [PROMPT, PROMPT], steering=[spec, False], use_tqdm=False,
            sampling_params=SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
        )
        assert outputs[0].outputs[0].finish_reason == "error"
        assert "conflict" in outputs[0].outputs[0].stop_reason
        assert outputs[1].outputs[0].finish_reason == "length"
        assert len(outputs[1].outputs[0].token_ids) == 4
        control = llm.generate(
            PROMPT, steering=steering_spec(scale=1.2, layers=[10]), use_tqdm=False,
            sampling_params=SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
        )[0]
        assert len(control.outputs[0].token_ids) == 2

    def test_two_vector_spec_steers(self, llm, plain, two_vector_spec):
        assert gen(llm, [PROMPT], two_vector_spec)[0] != plain

    def test_no_cross_request_contamination(self, llm, two_vector_spec):
        """Mechanism-level isolation for a [multi-vector, plain] batch:
        every applied position lies in rows routed to the apply's slot,
        and both sub-vectors fire on their layer ranges.
        """
        import os

        from helpers import read_trace

        trace_dir = os.environ["VLLM_STEER_TRACE_DIR"]
        steps_before, _ = read_trace(trace_dir, 0, ())
        start = max(steps_before, default=0)
        outs = gen(llm, [PROMPT, PROMPT], [two_vector_spec, None])
        assert outs[0] != outs[1], "multi-vector request did not steer"
        steps, applies = read_trace(trace_dir, start, (10, 18))
        assert applies, "mixed batch produced no steering applies"
        seen_layers = set()
        for rec in applies:
            step = steps[rec["step"]]
            qsl = step["query_start_loc"]
            slots = step["slots"]
            seen_layers.add(rec["layer"])
            for pos in rec["positions"]:
                req_idx = next(
                    i for i in range(len(qsl) - 1)
                    if qsl[i] <= pos < qsl[i + 1]
                )
                assert slots[req_idx] == rec["slot"], (
                    f"step {rec['step']}: position {pos} of slot "
                    f"{rec['slot']} lies in request {req_idx} routed to "
                    f"slot {slots[req_idx]}"
                )
        assert seen_layers == {10, 18}, (
            f"both sub-vectors must fire on their layer ranges, saw "
            f"{sorted(seen_layers)}"
        )


STEER_TEST_TRACE = True  # this module reads traces without using the fixture
REPRESENTATIVE_SCALES = [0.0, 1.0, 2.0, 5.0]
EXTENDED_SCALES = [round(i * 0.1, 1) for i in range(51)]


def _scale_spec(scale):
    return steering_spec(scale=scale, layers=LAYERS)


@pytest.fixture(scope="module")
def sweep(llm, request):
    import os

    from helpers import read_trace

    scales = (
        EXTENDED_SCALES if request.config.getoption("--steer-extended")
        else REPRESENTATIVE_SCALES
    )
    llm.preload_steer_vectors([DENSE_VECTOR])
    unsteered = gen(llm, [PROMPT], None, max_tokens=64)[0]
    minimum_distinct = 6 if len(scales) == len(EXTENDED_SCALES) else 2
    ref = {}
    for scale in scales:
        ref[scale] = gen(llm, [PROMPT], _scale_spec(scale), max_tokens=64)[0]
        if 0.0 in ref and len(set(ref.values())) >= minimum_distinct:
            break
    trace_dir = os.environ["VLLM_STEER_TRACE_DIR"]
    steps_before, _ = read_trace(trace_dir, 0, ())
    start = max(steps_before, default=0)
    batch = dict(zip(scales, gen(
        llm, [PROMPT] * len(scales),
        [_scale_spec(s) for s in scales], max_tokens=64,
    )))
    assert len(batch) == len(scales), "batched sweep lost requests"
    steps, applies = read_trace(trace_dir, start, (LAYERS[0],))
    return unsteered, ref, batch, steps, applies


class TestScaleSweep:
    def test_scale_zero_is_unsteered_and_sweep_varies(self, sweep):
        unsteered, ref, batch, _, _ = sweep
        assert ref[0.0] == unsteered, "scale 0.0 must equal the unsteered output"
        minimum_distinct = 6 if len(batch) == len(EXTENDED_SCALES) else 2
        assert len(set(ref.values())) >= minimum_distinct, "sweep outputs barely vary"

    def test_batched_sweep_produces_scale_dependent_outputs(self, sweep):
        _, _, batch, _, _ = sweep
        minimum_distinct = 6 if len(batch) == len(EXTENDED_SCALES) else 2
        assert len(set(batch.values())) >= minimum_distinct, (
            "batched sweep outputs barely vary across scales"
        )

    def test_batched_isolation_via_trace(self, llm, sweep):
        """Every applied position belongs to a request routed to that
        apply's slot — the mechanism-level no-cross-request guarantee."""
        _, _, batch, steps, applies = sweep
        capacity = (
            llm.llm_engine.vllm_config.steer_vector_config.max_steer_vectors
        )
        assert applies, "batched sweep produced no steering applies"
        distinct_slots = set()
        for rec in applies:
            step = steps[rec["step"]]
            qsl = step["query_start_loc"]
            slots = step["slots"]
            distinct_slots.add(rec["slot"])
            for pos in rec["positions"]:
                req_idx = next(
                    i for i in range(len(qsl) - 1)
                    if qsl[i] <= pos < qsl[i + 1]
                )
                assert slots[req_idx] == rec["slot"], (
                    f"step {rec['step']}: position {pos} of slot "
                    f"{rec['slot']} lies in request {req_idx} routed to "
                    f"slot {slots[req_idx]}"
                )
        # max_steer_vectors is a scheduling constraint: distinct
        # configs stream through a bounded, reused slot pool (all live
        # at once when capacity >= 51, in waves otherwise).
        assert distinct_slots and len(distinct_slots) <= min(
            capacity, len(batch)
        ), f"slot pool not bounded: {sorted(distinct_slots)}"
        assert all(0 <= s < capacity for s in distinct_slots), (
            f"slot ids escaped the capacity pool ({capacity}): "
            f"{sorted(distinct_slots)}"
        )
        for step_id, step in steps.items():
            live = {s for s in step["slots"] if s >= 0}
            assert len(live) <= capacity, (
                f"step {step_id} carried {len(live)} distinct configs "
                f"(> max_steer_vectors {capacity})"
            )
