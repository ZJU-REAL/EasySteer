# SPDX-License-Identifier: Apache-2.0
"""ApplySpec semantics against the steering trace, under chunked prefill.

One eager engine with 64-token chunks. The trace oracle yields exact
steered absolute positions, covering:
- phase selection ("prompt" covers every prompt token across chunks
  including 1-token tails; "generation" covers only decode steps);
- 1-token prompts classified correctly (the old length==1 heuristic's
  failure mode);
- exclusions composing with phase-wide selection;
- absolute/negative position filters and token filters (union);
- exact half-open generation windows;
- multi-vector specs with independent per-vector apply clauses.
"""

from vllm import SamplingParams

from helpers import DENSE_MODEL, DENSE_VECTOR, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    enforce_eager=True,
    enable_prefix_caching=False,
    enable_chunked_prefill=True,
    max_num_batched_tokens=64,
    max_num_seqs=4,
    max_model_len=512,
    gpu_memory_utilization=0.18,
)

PROMPT_LONG = list(range(100, 100 + 129))  # prefill chunks of 64, 64, 1
PROMPT_ONE = [100]
PARAMS = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)


class TestPhases:
    def test_prompt_phase_covers_chunked_prompt(self, trace):
        _, by_layer = trace.run(
            PROMPT_LONG, PARAMS, steering=steering_spec(phases=["prompt"])
        )
        pos = by_layer[10]
        assert {p for p, _ in pos} == set(range(129)), (
            "phases=['prompt'] must cover all 129 prompt tokens incl. the "
            "1-token tail chunk"
        )
        assert all(is_prefill for _, is_prefill in pos)

    def test_generation_phase_only_decode(self, trace):
        _, by_layer = trace.run(
            PROMPT_LONG, PARAMS, steering=steering_spec(phases=["generation"])
        )
        pos = by_layer[10]
        assert {p for p, _ in pos} == {129, 130, 131}
        assert all(not is_prefill for _, is_prefill in pos), (
            "the 1-token prompt tail chunk must not be steered as decode"
        )

    def test_one_token_prompt_prompt_phase(self, trace):
        assert trace.positions(
            PROMPT_ONE, PARAMS, steering=steering_spec(phases=["prompt"])
        ) == [0]

    def test_one_token_prompt_generation_phase(self, trace):
        assert trace.positions(
            PROMPT_ONE, PARAMS, steering=steering_spec(phases=["generation"])
        ) == [1, 2, 3]


class TestFilters:
    def test_exclusions_compose_with_phase_wide_selection(self, trace):
        assert trace.positions(
            PROMPT_LONG,
            PARAMS,
            steering=steering_spec(phases=["prompt"], exclude_positions=[0, -1]),
        ) == list(range(1, 128))

    def test_negative_position_fires_at_last_prompt_token(self, trace):
        assert trace.positions(
            PROMPT_LONG, PARAMS, steering=steering_spec(phases=["prompt"],
                                                        positions=[-1])
        ) == [128]

    def test_token_filter_matches_exact_positions(self, trace):
        assert trace.positions(
            PROMPT_LONG,
            PARAMS,
            steering=steering_spec(phases=["prompt"], tokens=[100, 150]),
        ) == [0, 50]

    def test_token_and_position_union_with_exclusion(self, trace):
        assert trace.positions(
            PROMPT_LONG,
            PARAMS,
            steering=steering_spec(
                phases=["prompt"],
                tokens=[100, 150],
                positions=[7],
                exclude_positions=[50],
            ),
        ) == [0, 7]


class TestGenerationWindow:
    def test_window_exact_first_two_decode_steps(self, trace):
        assert trace.positions(
            PROMPT_LONG,
            PARAMS,
            steering=steering_spec(phases=["generation"],
                                   generation_window=(0, 2)),
        ) == [129, 130]

    def test_window_skips_only_first_decode_step(self, trace):
        assert trace.positions(
            PROMPT_LONG,
            PARAMS,
            steering=steering_spec(phases=["generation"],
                                   generation_window=(1, None)),
        ) == [130, 131]


class TestMultiVector:
    def test_independent_apply_clauses_per_layer(self, trace):
        from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

        spec = SteeringSpec(vectors=[
            VectorSpec(source=DENSE_VECTOR, scale=0.5, layers=[10],
                       apply=ApplySpec(phases=["prompt"], positions=[0])),
            VectorSpec(source=DENSE_VECTOR, scale=0.5, layers=[12],
                       apply=ApplySpec(phases=["generation"])),
        ])
        _, by_layer = trace.run(PROMPT_LONG, PARAMS, layers=(10, 12),
                                steering=spec)
        assert sorted(p for p, _ in by_layer[10]) == [0]
        assert sorted(p for p, _ in by_layer[12]) == [129, 130, 131]
