"""Train and sample a LoReFT emoji intervention through EasySteer's shared helper.

Run from the repository root in the installed EasySteer environment:
    CUDA_VISIBLE_DEVICES=0 python easysteer/reft/pyreft/examples/basic_demo.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --save-dir results/emoji_loreft

REFT_MODEL_PATH can select a local copy of the same model. The helper's default
prompt template is the Qwen chat format. GPU selection stays with the caller.
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("REFT_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct"),
    )
    parser.add_argument("--save-dir", type=Path, default=Path("results/emoji_loreft"))
    parser.add_argument("--epochs", type=float, default=100)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs <= 0 or args.rank <= 0 or args.layer < 0:
        parser.error("epochs and rank must be positive, and layer must be non-negative")

    from easysteer.reft.train import EMOJI_EXAMPLES, generate_reft, train_reft

    model, tokenizer = train_reft(
        model_path=args.model,
        examples=EMOJI_EXAMPLES,
        intervention="loreft",
        layer=args.layer,
        low_rank_dimension=args.rank,
        device=args.device,
        num_train_epochs=args.epochs,
        save_dir=str(args.save_dir),
        output_dir=str(args.save_dir / "training"),
    )
    print(generate_reft(model, tokenizer, "Who are you?", device=args.device))
    print(
        f"LoReFT checkpoint saved to {args.save_dir}. Use algorithm='loreft' "
        "when loading it through easysteer.vectors.from_pyreft for vLLM inference."
    )


if __name__ == "__main__":
    main()
