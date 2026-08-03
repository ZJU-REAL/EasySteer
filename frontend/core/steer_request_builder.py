"""Build v2 ``SteeringSpec`` objects from the legacy frontend JSON fields.

The web UI still sends the v1 trigger fields (``prefill_trigger_tokens``,
``prefill_trigger_positions``, ``generate_trigger_tokens``). This module
translates them server-side into the v2 apply clauses (phases plus
token/position filters) so no JavaScript changes are needed.

Mapping implemented by :func:`build_apply_specs`:

- no trigger fields at all      -> phases=["prompt", "generation"], no filters
- prefill_trigger_tokens=[-1]   -> phases include "prompt", no token filter
- prefill_trigger_tokens=[ids]  -> phases=["prompt"], tokens=[ids]
- prefill_trigger_positions=[p] -> phases=["prompt"], positions=[p]
  (negative positions index from the end of the prompt, as in v1)
- generate_trigger_tokens=[-1]  -> phases include "generation", no token filter
- generate_trigger_tokens=[ids] -> phases=["generation"], tokens=[ids]

Prompt and generation clauses are merged into a single ``ApplySpec`` when
their filters coincide; otherwise one ``VectorSpec`` per phase is emitted
(the phases are disjoint, so the vectors never conflict).

A baseline (non-steered) generation is expressed as ``steering=None`` on
``llm.generate``; there is no baseline builder anymore.
"""

import logging
from typing import Any, Dict, List, Optional

from vllm.steer_vectors.api import ApplySpec, SteeringSpec, VectorSpec

logger = logging.getLogger(__name__)


def _vector_source_kwargs(algorithm: str, path: str) -> Dict[str, Any]:
    """File-format handling per algorithm.

    Data-only algorithms (linear, lm_steer, loreft) no longer accept a
    ``source`` path — the engine only loads its own formats. Interpret
    the user's file client-side with the easysteer payload adapters and
    pass the canonical payload via ``data=``.
    """
    if algorithm in ("linear", "lm_steer", "loreft"):
        import easysteer.vectors as vec

        adapter = {
            "linear": vec.from_linear_transport,
            "lm_steer": vec.from_lm_steer,
            "loreft": vec.from_pyreft,
        }[algorithm]
        return {"data": adapter(path)}
    return {"source": path}

# v1 sentinel inside a trigger-token list meaning "all tokens of this phase".
ALL_TOKENS_SENTINEL = -1


def _as_int_list(values: Optional[List[Any]]) -> Optional[List[int]]:
    """Normalize a UI-provided list; None and [] both mean "not set"."""
    if values is None or len(values) == 0:
        return None
    return [int(v) for v in values]


def build_apply_specs(
    prefill_trigger_tokens: Optional[List[int]] = None,
    prefill_trigger_positions: Optional[List[int]] = None,
    generate_trigger_tokens: Optional[List[int]] = None,
) -> List[ApplySpec]:
    """Translate the legacy v1 trigger fields into v2 apply clauses.

    Returns a non-empty list of ``ApplySpec`` clauses (see module docstring
    for the exact mapping). Multiple clauses are returned only when the
    prompt and generation filters cannot be expressed as one clause.
    """
    prefill_tokens = _as_int_list(prefill_trigger_tokens)
    positions = _as_int_list(prefill_trigger_positions)
    generate_tokens = _as_int_list(generate_trigger_tokens)

    if prefill_tokens is None and positions is None and generate_tokens is None:
        # v1 default: no triggers means steer every token of both phases.
        return [ApplySpec(phases=["prompt", "generation"])]

    want_prompt = prefill_tokens is not None or positions is not None
    if prefill_tokens is not None and ALL_TOKENS_SENTINEL in prefill_tokens:
        # -1 selects all prompt tokens; any other filters are subsumed.
        prefill_tokens = None
        positions = None

    want_generation = generate_tokens is not None
    if generate_tokens is not None and ALL_TOKENS_SENTINEL in generate_tokens:
        generate_tokens = None

    if (
        want_prompt
        and want_generation
        and positions is None
        and prefill_tokens == generate_tokens
    ):
        return [
            ApplySpec(phases=["prompt", "generation"], tokens=prefill_tokens)
        ]

    specs: List[ApplySpec] = []
    if want_prompt:
        specs.append(
            ApplySpec(phases=["prompt"], tokens=prefill_tokens, positions=positions)
        )
    if want_generation:
        specs.append(ApplySpec(phases=["generation"], tokens=generate_tokens))
    return specs


def build_single_vector_spec(
    vector_path: str,
    scale: float,
    target_layers: Optional[List[int]] = None,
    algorithm: str = "direct",
    name: Optional[str] = None,
    prefill_trigger_tokens: Optional[List[int]] = None,
    prefill_trigger_positions: Optional[List[int]] = None,
    generate_trigger_tokens: Optional[List[int]] = None,
    normalize: bool = False,
    debug: bool = False,
) -> SteeringSpec:
    """Build a single-vector ``SteeringSpec`` from the legacy request fields."""
    logger.info(
        "Building single-vector spec: name=%s, scale=%s, algorithm=%s, layers=%s",
        name, scale, algorithm, target_layers,
    )
    vectors = [
        VectorSpec(
            **_vector_source_kwargs(algorithm, vector_path),
            scale=scale,
            layers=target_layers or None,
            algorithm=algorithm,
            normalize=normalize,
            apply=apply_spec,
            name=name,
        )
        for apply_spec in build_apply_specs(
            prefill_trigger_tokens=prefill_trigger_tokens,
            prefill_trigger_positions=prefill_trigger_positions,
            generate_trigger_tokens=generate_trigger_tokens,
        )
    ]
    return SteeringSpec(vectors=vectors, debug=debug)


def build_multi_vector_spec(
    vector_configs: List[Dict[str, Any]],
    conflict_resolution: str = "sequential",
    debug: bool = False,
) -> SteeringSpec:
    """Build a multi-vector ``SteeringSpec`` from legacy per-vector dicts.

    Each dict may contain: path (required), scale, target_layers, algorithm,
    normalize, and the three legacy trigger fields.
    """
    logger.info(
        "Building multi-vector spec: conflict_resolution=%s, num_vectors=%d",
        conflict_resolution, len(vector_configs),
    )
    vectors: List[VectorSpec] = []
    for i, config in enumerate(vector_configs):
        path = config.get("path")
        if not path:
            raise ValueError(f'Vector config {i + 1} is missing required "path" field')
        for apply_spec in build_apply_specs(
            prefill_trigger_tokens=config.get("prefill_trigger_tokens"),
            prefill_trigger_positions=config.get("prefill_trigger_positions"),
            generate_trigger_tokens=config.get("generate_trigger_tokens"),
        ):
            algorithm = config.get("algorithm", "direct")
            vectors.append(
                VectorSpec(
                    **_vector_source_kwargs(algorithm, path),
                    scale=config.get("scale", 1.0),
                    layers=config.get("target_layers") or None,
                    algorithm=algorithm,
                    normalize=config.get("normalize", False),
                    apply=apply_spec,
                )
            )
    return SteeringSpec(
        vectors=vectors, conflict=conflict_resolution, debug=debug
    )
