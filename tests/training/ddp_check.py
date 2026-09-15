# SPDX-License-Identifier: Apache-2.0
"""Compare native DDP updates with the same global batches on one device.

Run ``--mode reference`` with one visible GPU, then ``--mode distributed``
under ``torchrun --standalone --nproc_per_node=2`` with both GPUs visible.
CPU tests run the same comparison with gloo and ``--device cpu``.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TrainerCallback,
    set_seed,
)

from easysteer.training import load_checkpoint, train
from easysteer.training.model import SteeringModel

EXAMPLES = [
    ["a", "b"],
    ["c d", "e f g h i"],
    ["d e f", "a b"],
    ["g", "a b c d e f g h"],
    ["h i", "c"],
    ["a b c", "d e f g"],
    ["f g", "h i a b c d"],
    ["i", "e f"],
]
PROMPT_TOKENS = [3, 5, 10, 11]  # a, c, h, i; three prompts contain none.


def prepare_model(path):
    set_seed(321)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=2,
            use_cache=False,
        )
    )
    model.save_pretrained(path)
    vocabulary = ["<pad>", "<unk>", "<eos>", *"abcdefghi"]
    tokenizer = Tokenizer(
        WordLevel({token: index for index, token in enumerate(vocabulary)}, "<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(path)


def parameters(model):
    return torch.cat([p.detach().flatten() for p in model.adapter.parameters()])


def run(args):
    torch.set_num_threads(1)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    model_path = args.output / "model"
    if args.mode == "reference":
        prepare_model(model_path)
    snapshots = []
    save_calls = []
    selections = []
    forward_hooks = []
    original_save = SteeringModel.save

    def record_save(model, path):
        save_calls.append(str(path))
        return original_save(model, path)

    SteeringModel.save = record_save

    def record_selection(model, inputs, kwargs):
        ids = kwargs["input_ids"]
        prompt = torch.arange(ids.shape[1], device=ids.device)[None, :]
        prompt = prompt < kwargs["prompt_lengths"][:, None]
        matches = torch.zeros_like(ids, dtype=torch.bool)
        for token in PROMPT_TOKENS:
            matches |= ids == token
        selections.append((prompt & matches).any(dim=1).cpu().tolist())

    class CaptureUpdates(TrainerCallback):
        def snapshot(self, model):
            values = parameters(model)
            if dist.is_initialized():
                ranks = [torch.empty_like(values) for _ in range(dist.get_world_size())]
                dist.all_gather(ranks, values)
                for other in ranks:
                    torch.testing.assert_close(values, other, atol=0, rtol=0)
            snapshots.append(values.cpu())

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.snapshot(model)
            forward_hooks.append(
                model.register_forward_pre_hook(record_selection, with_kwargs=True)
            )

        def on_step_end(self, args, state, control, model=None, **kwargs):
            self.snapshot(model)

    run_dir = args.output / args.mode
    distributed = args.mode == "distributed"
    wrapper, _ = train(
        str(model_path),
        EXAMPLES,
        algorithm=args.algorithm,
        device=args.device,
        layer=0,
        rank=4,
        apply={"prompt_tokens": PROMPT_TOKENS},
        prompt_template="%s",
        output_dir=str(run_dir / "trainer"),
        save_dir=run_dir / "adapter",
        callbacks=[CaptureUpdates()],
        max_steps=2,
        per_device_train_batch_size=1 if distributed else 4,
        gradient_accumulation_steps=2 if distributed else 1,
        learning_rate=0.2,
        optim="sgd",
        lr_scheduler_type="constant",
        max_grad_norm=0,
        save_strategy="steps",
        save_steps=1,
        logging_strategy="no",
        disable_tqdm=True,
        seed=17,
        data_seed=23,
        ddp_backend=("gloo" if args.device == "cpu" else "nccl")
        if distributed
        else None,
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    for hook in forward_hooks:
        hook.remove()
    assert len(snapshots) == 3
    assert not torch.equal(snapshots[0], snapshots[-1])
    assert len(save_calls) == (3 if rank == 0 else 0), (rank, save_calls)
    assert all(p.grad is None for p in wrapper.base_model.parameters())
    exported = load_checkpoint(run_dir / "adapter")
    assert exported.payload.to_wire() == wrapper.to_payload().to_wire()
    if distributed:
        assert dist.get_world_size() == 2
        gathered_selections = [None, None]
        dist.all_gather_object(gathered_selections, selections)
        all_selections = [
            batch for rank_batches in gathered_selections for batch in rank_batches
        ]
        assert sum(not any(batch) for batch in all_selections) == 3
        expected = torch.load(args.output / "reference.pt", weights_only=True)
        torch.testing.assert_close(snapshots[0], expected[0], atol=0, rtol=0)
        relative_errors = []
        for actual, baseline in zip(snapshots[1:], expected[1:], strict=True):
            update = baseline - expected[0]
            error = (actual - baseline).norm() / update.norm()
            relative_errors.append(error.item())
            # Compare changes, not large initial weights, so an incorrect
            # per-rank loss average cannot hide behind absolute tolerances.
            tolerance = 1e-4 if args.device == "cpu" else 0.02
            assert error < tolerance, (args.algorithm, error.item(), tolerance)
    else:
        torch.save(snapshots, args.output / "reference.pt")
    report = {
        "algorithm": args.algorithm,
        "device": str(next(wrapper.base_model.parameters()).device),
        "rank": rank,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "save_calls": len(save_calls),
        "steps": len(snapshots) - 1,
        "selected_examples_by_microbatch": selections,
        "update_max_abs": (snapshots[-1] - snapshots[0]).abs().max().item(),
    }
    if distributed:
        report["reference_update_relative_errors"] = relative_errors
        report["reference_max_abs_error"] = max(
            (actual - baseline).abs().max().item()
            for actual, baseline in zip(snapshots, expected, strict=True)
        )
    (run_dir / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "distributed"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--algorithm", choices=("direct", "loreft"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
