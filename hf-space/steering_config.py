"""Convert the Space's form/preset fields into canonical steering specs.

This module does no I/O and imports no UI, tensor or engine dependencies.
The caller supplies path/payload resolution for its execution environment.
"""


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated layer, position or token fields."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _apply_clause(prefill_tokens, prefill_positions, generate_tokens):
    """Translate form trigger fields into one canonical apply clause.

    -1 in a token list means "every token of that phase" (prompt="all" /
    generation="all"); no fields at all means both phases whole. Phase
    coverage is independent per phase, so every combination fits a
    single clause.
    """
    want_prompt = prefill_tokens is not None or prefill_positions is not None
    want_generation = generate_tokens is not None
    if not want_prompt and not want_generation:
        return {"prompt": "all", "generation": "all"}

    clause = {}
    if want_prompt:
        if prefill_tokens is not None and -1 in prefill_tokens:
            clause["prompt"] = "all"
        else:
            if prefill_tokens:
                clause["prompt_tokens"] = prefill_tokens
            if prefill_positions:
                clause["prompt_positions"] = prefill_positions
    if want_generation:
        if -1 in generate_tokens:
            clause["generation"] = "all"
        elif generate_tokens:
            clause["generation_tokens"] = generate_tokens
    return clause


def _vector_wire(config, resolve_source, scale_override=None):
    algorithm = config.get("algorithm", "direct")
    wire = {
        **resolve_source(algorithm, config["path"], config.get("payload_path")),
        "scale": float(config.get("scale", 1.0)) if scale_override is None else scale_override,
        "algorithm": algorithm,
        "normalize": bool(config.get("normalize", False)),
    }

    def field(name):
        value = config.get(name)
        return parse_int_list(str(value)) if value is not None and str(value).strip() else None

    if layers := field("target_layers"):
        wire["layers"] = layers
    wire["apply"] = _apply_clause(
        field("prefill_trigger_tokens"),
        field("prefill_trigger_positions"),
        field("generate_trigger_tokens"),
    )
    return wire


def build_single_spec_wire(config, resolve_source, scale_override=None) -> dict:
    """Build a single-vector spec using the caller's source resolver."""
    return {"vectors": [_vector_wire(config["steer_vector"], resolve_source, scale_override)]}


def build_multi_spec_wire(config, resolve_source) -> dict:
    """Build a multi-vector spec, preserving configured order and conflict handling."""
    return {
        "vectors": [_vector_wire(vector, resolve_source) for vector in config["vector_configs"]],
        "conflict": config["steer_vector"].get("conflict_resolution", "sequential"),
    }
