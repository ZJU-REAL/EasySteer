"""CPU-only configuration and payload loading for the Space."""

import json
import os
from copy import deepcopy
from pathlib import Path

ALGORITHM_CAPABILITIES = json.loads(
    Path(__file__).with_name("algorithm_capabilities.json").read_text()
)


def demo_mode(environ=None):
    environ = os.environ if environ is None else environ
    mode = environ.get("DEMO_MODE", "api").strip().lower()
    if mode not in {"api", "gpu"}:
        raise ValueError("DEMO_MODE must be api or gpu")
    if mode == "api":
        missing = [
            key
            for key in ("VLLM_API_URL", "VLLM_MODEL_NAME")
            if not environ.get(key, "").strip()
        ]
        if missing:
            raise ValueError("API mode requires " + ", ".join(missing))
    return mode


def load_payload(path, algorithm):
    """Read an exported canonical payload without importing torch or vLLM."""
    expected_kind = ALGORITHM_CAPABILITIES[algorithm]["payload_kind"]
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Payload file missing: {path}; run export_payload.py first")
    with path.open() as handle:
        payload = json.load(handle)
    if payload.get("version") != 1 or payload.get("kind") != expected_kind:
        raise ValueError(f"{path} is not a version 1 {algorithm} payload")
    tensors = payload.get("tensors")
    data = (
        payload.get("extra", {}).get("layers") if expected_kind == "router" else tensors
    )
    if not isinstance(tensors, dict) or not data or not payload.get("sha256"):
        raise ValueError(f"{path} lacks canonical payload data or its content hash")
    return payload


def load_steering_spec(config, resolve_source, *, app_dir, scale_override=None):
    """Resolve preset file references without mutating the configuration."""
    spec = deepcopy(config["steering"])
    for vector in spec["vectors"]:
        if "payload_path" in vector:
            vector["data"] = load_payload(
                Path(app_dir) / vector.pop("payload_path"), vector["algorithm"]
            )
        if "source" in vector:
            vector["source"] = resolve_source(vector["source"])
    if scale_override is not None:
        spec["vectors"][0]["scale"] = scale_override
    return spec
