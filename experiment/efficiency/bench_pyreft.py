# SPDX-License-Identifier: Apache-2.0
"""PyReFT-compatible additive intervention benchmark.

The comparison attaches a zero additive vector to every layer. The default
benchmark uses padded batches of 256 prompts. Use ``--samples`` to submit
more prompts in successive batches, or pass ``--batch 0`` for sequential mode.
"""

import argparse
import time

from common import MODEL, N_SEQUENTIAL, load_examples, nonnegative_int, report


def load_reft_model(device):
    import torch
    import transformers

    from easysteer.reft import pyreft
    from easysteer.reft.pyreft.reft.algorithms import BiasIntervention

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=device
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL, padding_side="left", use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Use the same additive h + vector operation as EasySteer's direct
    # algorithm. A zero bias keeps the generated tokens unchanged while the
    # intervention wrapper remains active on every layer.
    reft_config = pyreft.ReftConfig(
        representations=[
            {
                "layer": layer,
                "component": "block_output",
                "intervention": BiasIntervention(
                    embed_dim=model.config.hidden_size,
                ),
            }
            for layer in range(model.config.num_hidden_layers)
        ]
    )
    reft_model = pyreft.get_reft_model(model, reft_config)
    with torch.no_grad():
        for module in reft_model.interventions.values():
            for parameter in module.parameters():
                parameter.zero_()
    reft_model.set_device(device)
    reft_model.eval()
    return reft_model, tokenizer


def generated_token_count(generated, input_width):
    """Count the generated suffix after the padded input in fixed-length runs."""
    return generated.shape[0] * (generated.shape[1] - input_width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        type=nonnegative_int,
        default=256,
        help="per-call batch size; 0 = sequential (paper: 256)",
    )
    parser.add_argument(
        "--samples",
        type=nonnegative_int,
        default=None,
        help="total prompts to evaluate; defaults to --batch",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, choices=[128, 2048])
    args = parser.parse_args()
    if args.batch and args.samples == 0:
        parser.error("--samples must be positive when --batch is nonzero")

    import torch

    device = "cuda"
    reft_model, tokenizer = load_reft_model(device)
    gen_kwargs = {
        "intervene_on_prompt": False,
        "max_new_tokens": args.max_tokens,
        "min_new_tokens": args.max_tokens,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
    }
    warmup_kwargs = {**gen_kwargs, "min_new_tokens": 8, "max_new_tokens": 8}

    if args.batch:
        samples = args.samples if args.samples is not None else args.batch
        examples = load_examples(samples)
        batches = []
        for start in range(0, samples, args.batch):
            inputs = tokenizer(
                examples[start : start + args.batch],
                return_tensors="pt",
                padding=True,
            )
            batches.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )

        warmup_batch = {key: value.to(device) for key, value in batches[0].items()}
        reft_model.generate(warmup_batch, **warmup_kwargs)
        del warmup_batch
        torch.cuda.synchronize()
        start = time.perf_counter()
        tokens = 0
        for input_dict in batches:
            input_dict = {key: value.to(device) for key, value in input_dict.items()}
            _, generated = reft_model.generate(input_dict, **gen_kwargs)
            tokens += generated_token_count(
                generated, input_dict["input_ids"].shape[1]
            )
            del input_dict, generated
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        report(tokens, elapsed, samples)
    else:
        examples = load_examples(N_SEQUENTIAL)
        prepared = [tokenizer(e, return_tensors="pt").to(device) for e in examples]
        tokens = 0
        reft_model.generate(
            {key: prepared[0][key] for key in ("input_ids", "attention_mask")},
            **warmup_kwargs,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        for inputs in prepared:
            input_dict = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            _, generated = reft_model.generate(input_dict, **gen_kwargs)
            tokens += generated_token_count(generated, inputs["input_ids"].shape[1])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        report(tokens, elapsed, len(examples))


if __name__ == "__main__":
    main()
