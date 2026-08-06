# SPDX-License-Identifier: Apache-2.0
"""ApplySpec semantics against the steering trace, under chunked prefill.

One eager engine with 64-token chunks. The trace oracle yields exact
steered absolute positions, covering:
- phase selection ("prompt" covers every prompt token across chunks
  including 1-token tails; "generation" covers only decode steps);
- 1-token prompts classified correctly (the old length==1 heuristic's
  failure mode);
- exclusions composing with phase-wide selection;
- include selectors (tokens, positions, prompt windows, generation
  positions/windows) unioning within the phase gate, exclude twins
  always vetoing;
- exact half-open generation windows;
- multi-vector specs with independent per-vector apply clauses.
"""

from vllm import SamplingParams

from helpers import DENSE_MODEL, DENSE_VECTOR, steering_spec

ENGINE_KWARGS = dict(
    model=DENSE_MODEL,
    enable_steer_vector=True,
    steer_algorithms=["direct"],
    steer_multi_vector=True,
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
            PROMPT_LONG, PARAMS, steering=steering_spec(prompt="all")
        )
        pos = by_layer[10]
        assert {p for p, _ in pos} == set(range(129)), (
            "prompt='all' must cover all 129 prompt tokens incl. the "
            "1-token tail chunk"
        )
        assert all(is_prefill for _, is_prefill in pos)

    def test_generation_phase_only_decode(self, trace):
        _, by_layer = trace.run(
            PROMPT_LONG, PARAMS, steering=steering_spec(generation="all")
        )
        pos = by_layer[10]
        assert {p for p, _ in pos} == {129, 130, 131}
        assert all(not is_prefill for _, is_prefill in pos), (
            "the 1-token prompt tail chunk must not be steered as decode"
        )

    def test_one_token_prompt_prompt_phase(self, trace):
        assert trace.positions(
            PROMPT_ONE, PARAMS, steering=steering_spec(prompt="all")
        ) == [0]

    def test_one_token_prompt_generation_phase(self, trace):
        assert trace.positions(
            PROMPT_ONE, PARAMS, steering=steering_spec(generation="all")
        ) == [1, 2, 3]


class TestFilters:
    def test_exclusions_compose_with_phase_wide_selection(self, trace):
        assert trace.positions(
            PROMPT_LONG,
            PARAMS,
            steering=steering_spec(prompt="all", exclude_prompt_positions=[0, -1]),
        ) == list(range(1, 128))

    def test_negative_position_fires_at_last_prompt_token(self, trace):
        assert trace.positions(
            PROMPT_LONG, PARAMS, steering=steering_spec(
                                                        prompt_positions=[-1])
        ) == [128]

    def test_token_filter_matches_exact_positions(self, trace):
        assert trace.positions(
            PROMPT_LONG,
            PARAMS,
            steering=steering_spec(prompt_tokens=[100, 150]),
        ) == [0, 50]

    def test_token_and_position_union_with_exclusion(self, trace):
        assert trace.positions(
            PROMPT_LONG,
            PARAMS,
            steering=steering_spec(
                prompt_tokens=[100, 150],
                prompt_positions=[7],
                exclude_prompt_positions=[50],
            ),
        ) == [0, 7]


class TestGenerationWindow:
    def test_window_exact_first_two_decode_steps(self, trace):
        assert trace.positions(
            PROMPT_LONG,
            PARAMS,
            steering=steering_spec(
                                   generation_window=(0, 2)),
        ) == [129, 130]

    def test_window_skips_only_first_decode_step(self, trace):
        assert trace.positions(
            PROMPT_LONG,
            PARAMS,
            steering=steering_spec(
                                   generation_window=(1, None)),
        ) == [130, 131]


class TestMultiVector:
    def test_independent_apply_clauses_per_layer(self, trace):
        from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

        spec = SteeringSpec(vectors=[
            VectorSpec(source=DENSE_VECTOR, scale=0.5, layers=[10],
                       apply=ApplySpec(prompt_positions=[0])),
            VectorSpec(source=DENSE_VECTOR, scale=0.5, layers=[12],
                       apply=ApplySpec(generation="all")),
        ])
        _, by_layer = trace.run(PROMPT_LONG, PARAMS, layers=(10, 12),
                                steering=spec)
        assert sorted(p for p, _ in by_layer[10]) == [0]
        assert sorted(p for p, _ in by_layer[12]) == [129, 130, 131]


class TestBatchedPerRequestPositions:
    """Adversarial continuous batching: six requests with different
    prompt lengths, different position specs, and different generation
    lengths submitted in one call, with max_num_seqs=4 forcing
    scheduling waves and 64-token chunks interleaving prefill chunks
    with other requests' decode steps. The trace must attribute every
    steered position to the right request, and each request's absolute
    positions must match its own spec exactly."""

    CASES = [
        # (prompt_len, max_tokens, apply_kwargs or None, expected_fn)
        (129, 4, dict(prompt_positions=[-1]),
         lambda L, mt: {L - 1}),
        (37, 6, dict(prompt_positions=[0]),
         lambda L, mt: {0}),
        (1, 5, dict(generation="all"),
         lambda L, mt: set(range(L, L + mt - 1))),
        (61, 6, dict(generation_window=(1, 3)),
         lambda L, mt: {L + 1, L + 2}),
        # Include selectors union: the windowed decode step joins the
        # positions matches instead of being vetoed by them
        # (clause.py union semantics).
        (45, 4, dict(prompt_positions=[-2, -1],
                     generation_window=(0, 1)),
         lambda L, mt: {L - 2, L - 1, L}),
        # The window is an include selector: once present, only its
        # matches are selected — prompt tokens need their own selector
        # (prompt_window) to join a cross-phase clause.
        (33, 4, dict(
                     prompt_window=(0, None),
                     generation_window=(0, 1)),
         lambda L, mt: set(range(L)) | {L}),
        # New symmetric selectors: the prompt tail via a negative-bound
        # prompt window, exact decode steps via generation_positions,
        # with an exclude twin vetoing one of each.
        (25, 5, dict(
                     prompt_window=(-3, None),
                     generation_positions=[0, 2],
                     exclude_prompt_window=(-2, -1),
                     exclude_generation_positions=[2]),
         lambda L, mt: {L - 3, L - 1, L}),
        (80, 4, None, lambda L, mt: set()),
    ]

    def test_every_request_steers_its_own_positions(self, trace):
        from vllm.inputs import TokensPrompt

        from helpers import read_trace

        prompts, params, steering, expected = [], [], [], []
        for i, (length, mt, apply_kwargs, expect) in enumerate(self.CASES):
            prompts.append(TokensPrompt(
                prompt_token_ids=list(range(100 + i, 100 + i + length))
            ))
            params.append(SamplingParams(temperature=0, max_tokens=mt,
                                         ignore_eos=True))
            steering.append(
                None if apply_kwargs is None
                else steering_spec(scale=0.5 + 0.01 * i, **apply_kwargs)
            )
            expected.append(expect(length, mt))

        start = trace._max_step()
        outs = trace.llm.generate(prompts, params, steering=steering,
                                  use_tqdm=False)
        steps, applies = read_trace(trace.trace_dir, start, (10,))

        per_req = {out.request_id: set() for out in outs}
        for rec in applies:
            step = steps[rec["step"]]
            qsl = step["query_start_loc"]
            for pos in rec["positions"]:
                ri = next(i for i in range(len(qsl) - 1)
                          if qsl[i] <= pos < qsl[i + 1])
                # Engine req ids carry a uniquifying suffix
                # ("11-893e12e8"); RequestOutput.request_id is the prefix.
                rid = step["req_ids"][ri].split("-", 1)[0]
                assert rid in per_req, f"trace names unknown request {rid}"
                per_req[rid].add(pos - qsl[ri] + step["num_computed"][ri])

        for i, out in enumerate(outs):
            got = per_req[out.request_id]
            assert got == expected[i], (
                f"request {i} (len={self.CASES[i][0]}, "
                f"spec={self.CASES[i][2]}): steered positions {sorted(got)} "
                f"!= expected {sorted(expected[i])}"
            )
