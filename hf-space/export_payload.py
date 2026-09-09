"""Export a trusted checkpoint using the installed EasySteer adapters.

Run in an EasySteer/vllm-steer environment; the resulting JSON is usable by
the lightweight API Space without torch or vLLM installed.
"""

import argparse
import json
from pathlib import Path


def export_payload(checkpoint, algorithm, output):
    checkpoint = Path(checkpoint)
    files = checkpoint.glob("*.bin") if checkpoint.is_dir() else [checkpoint]
    for path in files:
        with path.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(
                    f"{path} is a Git LFS pointer. Download the actual checkpoint "
                    "from the repository's file page before exporting."
                )

    import easysteer.vectors as vec

    adapter = {
        "linear": vec.from_linear_transport,
        "lm_steer": vec.from_lm_steer,
        "loreft": vec.from_pyreft,
    }[algorithm]
    wire = vec.to_json_payload(adapter(str(checkpoint)))
    if algorithm == "loreft" and wire["kind"] != "reft":
        raise ValueError("This is a bias checkpoint, not a LoReFT checkpoint")
    Path(output).write_text(json.dumps(wire, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--algorithm", choices=["linear", "lm_steer", "loreft"], required=True)
    args = parser.parse_args()
    export_payload(args.checkpoint, args.algorithm, args.output)
