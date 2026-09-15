"""Export a trusted checkpoint using the installed EasySteer adapters.

Run in an EasySteer/vllm-steer environment; the resulting JSON is usable by
the lightweight API Space without torch or vLLM installed.
"""

import argparse
import json
from pathlib import Path


def export_payload(checkpoint, algorithm, output, *, format="legacy"):
    checkpoint = Path(checkpoint)
    files = checkpoint.glob("*.bin") if checkpoint.is_dir() else [checkpoint]
    for path in files:
        with path.open("rb") as handle:
            if handle.read(80).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise ValueError(
                    f"{path} is a Git LFS pointer. Download the actual checkpoint "
                    "from the repository's file page before exporting."
                )

    import easysteer.vectors as vec

    if format == "training":
        adapter = vec.from_training
    elif format == "legacy":
        adapters = {
            "linear": vec.from_linear_transport,
            "lm_steer": vec.from_lm_steer,
            "loreft": vec.from_pyreft,
        }
        if algorithm not in adapters:
            raise ValueError("direct checkpoints require --format training")
        adapter = adapters[algorithm]
    else:
        raise ValueError(f"Unknown checkpoint format: {format!r}")
    wire = vec.to_json_payload(adapter(str(checkpoint)))
    expected_kind = {
        "direct": "direction",
        "linear": "linear",
        "lm_steer": "lowrank",
        "loreft": "reft",
    }[algorithm]
    if wire["kind"] != expected_kind:
        raise ValueError(f"Checkpoint payload does not support {algorithm}")
    Path(output).write_text(json.dumps(wire, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--algorithm", choices=["direct", "linear", "lm_steer", "loreft"], required=True
    )
    parser.add_argument("--format", choices=["legacy", "training"], default="legacy")
    args = parser.parse_args()
    export_payload(args.checkpoint, args.algorithm, args.output, format=args.format)
