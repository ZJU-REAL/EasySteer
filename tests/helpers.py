# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the EasySteer validation suites.

The steering trace (VLLM_STEER_TRACE_DIR) is the exact oracle used by
mechanism-level tests: it records, per engine step, the batch geometry
and the flat positions each steered layer applied to. `TraceOracle`
wraps generate() and returns the steered absolute positions.
"""

import glob
import json
import os

DENSE_MODEL = os.environ.get(
    "STEER_TEST_MODEL", "/data/zju-130/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"
)
DENSE_VECTOR = os.path.expanduser(
    os.environ.get("STEER_TEST_VECTOR", "~/EasySteer/vectors/happy_diffmean.gguf")
)
MOE_MODEL = os.path.expanduser(
    os.environ.get("STEER_TEST_MOE_MODEL", "~/models/OLMoE-1B-7B-0125-Instruct")
)
QWEN3_MODEL = os.environ.get(
    "STEER_TEST_QWEN3", "/data/zju-130/shenyl/hf/model/Qwen/Qwen3-30B-A3B"
)


_INCLUDE_KWARGS = (
    "prompt", "generation", "prompt_tokens", "prompt_positions",
    "prompt_window", "generation_tokens", "generation_positions",
    "generation_window",
)


def steering_spec(source=DENSE_VECTOR, scale=0.5, layers=(10,),
                  algorithm="direct", normalize=False, params=None,
                  conflict="priority", extra_vectors=(), **apply_kwargs):
    """Build a single-vector SteeringSpec (the common test shape).

    With no include selector among apply_kwargs the clause covers both
    phases whole, so exclude-only callers keep the old default scope.
    """
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    if not any(k in apply_kwargs for k in _INCLUDE_KWARGS):
        apply_kwargs["prompt"] = "all"
        apply_kwargs["generation"] = "all"
    vec = VectorSpec(
        source=source,
        algorithm=algorithm,
        scale=scale,
        layers=list(layers) if layers is not None else None,
        normalize=normalize,
        params=dict(params or {}),
        apply=ApplySpec(**apply_kwargs),
    )
    return SteeringSpec(vectors=[vec, *extra_vectors], conflict=conflict)


def read_trace(trace_dir, min_step, layers):
    """Trace records with step id > min_step; applies filtered to layers."""
    steps, applies = {}, []
    for path in glob.glob(os.path.join(trace_dir, "*.jsonl")):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["step"] <= min_step:
                    continue
                if rec["type"] == "step":
                    steps[rec["step"]] = rec
                elif rec["type"] == "apply" and rec["layer"] in layers:
                    applies.append(rec)
    return steps, applies


class TraceOracle:
    """Generate with steering and report steered absolute positions.

    Single-request batches only (asserts): positions are offset by the
    request's computed-token count at each step, so results are exact
    absolute sequence positions regardless of chunking.
    """

    def __init__(self, llm, trace_dir):
        self.llm = llm
        self.trace_dir = trace_dir

    def _max_step(self):
        steps, _ = read_trace(self.trace_dir, 0, ())
        return max(steps, default=0)

    def run(self, prompt_ids, sampling_params, layers=(10,), **gen_kwargs):
        """Returns (output, {layer: [(abs_pos, is_prefill_step), ...]})."""
        from vllm.inputs import TokensPrompt

        start = self._max_step()
        out = self.llm.generate(
            TokensPrompt(prompt_token_ids=list(prompt_ids)),
            sampling_params,
            use_tqdm=False,
            **gen_kwargs,
        )[0]
        steps, applies = read_trace(self.trace_dir, start, layers)
        by_layer = {layer: [] for layer in layers}
        for rec in applies:
            step = steps[rec["step"]]
            assert len(step["req_ids"]) == 1, "oracle expects single-request batches"
            num_computed = step["num_computed"][0]
            is_prefill = step["num_output"][0] == 0
            for pos in rec["positions"]:
                by_layer[rec["layer"]].append((pos + num_computed, is_prefill))
        return out, by_layer

    def positions(self, prompt_ids, sampling_params, layer=10, **gen_kwargs):
        """Sorted steered absolute positions for one layer."""
        _, by_layer = self.run(
            prompt_ids, sampling_params, layers=(layer,), **gen_kwargs
        )
        return sorted(p for p, _ in by_layer[layer])
