# SPDX-License-Identifier: Apache-2.0
"""Reproduce the LoReFT emoji experiment with native EasySteer training."""

import argparse
import json
import os
from pathlib import Path

DATA = Path(__file__).with_name("training_examples.json")


def examples():
    return json.loads(DATA.read_text(encoding="utf-8"))


def train_adapter(args):
    import torch.distributed as dist
    from transformers import TrainerCallback, set_seed

    from easysteer.training import train

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch_size % world_size:
        raise ValueError(
            "global batch size must be divisible by the training world size"
        )
    records = examples()
    history = []

    class LogCallback(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            if logs:
                history.append({"step": state.global_step, **logs})

    set_seed(args.seed)
    train(
        args.model,
        [[item["instruction"], item["emoji"]] for item in records],
        algorithm="loreft",
        component="hidden_states",
        layer=8,
        rank=4,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.global_batch_size // world_size,
        learning_rate=4e-3,
        seed=args.seed,
        logging_steps=20,
        disable_tqdm=True,
        report_to=[],
        save_strategy="no",
        output_dir=str(args.output_dir / "trainer"),
        save_dir=args.output_dir,
        callbacks=[LogCallback()],
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        result = {
            "model": args.model,
            "seed": args.seed,
            "epochs": args.epochs,
            "global_batch_size": args.global_batch_size,
            "world_size": world_size,
            "examples": len(records),
            "history": history,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "training.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def evaluate_adapter(args):
    from vllm import LLM, SamplingParams
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    from easysteer.training import load_checkpoint
    from easysteer.vectors import from_pyreft

    checkpoint = load_checkpoint(args.checkpoint)
    records = examples()
    prompts = [
        checkpoint.config.prompt_template % row["instruction"] for row in records
    ]
    specs = {"baseline": False, "native": checkpoint.to_spec()}
    for index, path in enumerate(args.compare_checkpoint):
        other = load_checkpoint(path)
        if other.config != checkpoint.config:
            raise ValueError(
                "comparison checkpoints must share the training configuration"
            )
        specs[f"comparison_{index}"] = other.to_spec()
    if args.legacy_checkpoint:
        specs["legacy"] = SteeringSpec(
            vectors=[
                VectorSpec(
                    data=from_pyreft(str(args.legacy_checkpoint)),
                    algorithm="loreft",
                    apply=ApplySpec(prompt_positions=[-1]),
                )
            ]
        )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enable_steer_vector=True,
        steering_config=checkpoint.to_spec().model_dump_json(),
        enforce_eager=args.eager,
        attention_backend="TRITON_ATTN",
        enable_prefix_caching=False,
        max_model_len=512,
        max_num_seqs=16,
        gpu_memory_utilization=0.4,
    )
    sampling = SamplingParams(temperature=0, max_tokens=256)
    results = {}
    for name, spec in specs.items():
        outputs = llm.generate(prompts, sampling, steering=spec, use_tqdm=False)
        rows = [
            {
                "instruction": row["instruction"],
                "expected": row["emoji"],
                "text": output.outputs[0].text,
                "token_ids": output.outputs[0].token_ids,
                "exact_match": output.outputs[0].text.strip() == row["emoji"],
            }
            for row, output in zip(records, outputs)
        ]
        results[name] = {
            "exact_matches": sum(row["exact_match"] for row in rows),
            "total": len(rows),
            "outputs": rows,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "tp": args.tp,
        "checkpoint": str(args.checkpoint),
        "comparisons": [str(path) for path in args.compare_checkpoint],
        "results": results,
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    # The notebook demonstrates these two exact instruction-to-emoji answers.
    for name, result in results.items():
        if name in {"baseline", "legacy"}:
            continue
        if not all(row["exact_match"] for row in result["outputs"][:2]):
            raise AssertionError(
                f"{name} did not reproduce the notebook's two emoji answers"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--epochs", type=float, default=200)
    training.add_argument("--global-batch-size", type=int, default=10)
    training.add_argument("--seed", type=int, default=42)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument(
        "--compare-checkpoint", type=Path, action="append", default=[]
    )
    evaluation.add_argument("--legacy-checkpoint", type=Path)
    evaluation.add_argument("--tp", type=int, default=1)
    evaluation.add_argument("--eager", action="store_true")
    for subparser in (training, evaluation):
        subparser.add_argument("--model", required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "train" and (args.epochs <= 0 or args.global_batch_size <= 0):
        parser.error("epochs and global batch size must be positive")
    if args.stage == "evaluate" and args.tp <= 0:
        parser.error("tensor parallel size must be positive")
    (train_adapter if args.stage == "train" else evaluate_adapter)(args)


if __name__ == "__main__":
    main()
