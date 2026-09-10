# SPDX-License-Identifier: Apache-2.0
"""pyreft (HF transformers) efficiency benchmark (EasySteer paper, 5.1).

The framework comparison attaches rank-4 LoReFT to every layer and zeros its
parameters. Sequential by default; --batch N (paper: 256) times one padded
batch instead.
"""

import argparse
import time

from common import MODEL, N_SEQUENTIAL, load_examples, nonnegative_int, report


def load_reft_model(device):
    import torch
    import transformers

    from easysteer.reft import pyreft

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=device
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL, padding_side="left", use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Retain the all-layer LoReFT computation with zeroed parameters.
    reft_config = pyreft.ReftConfig(
        representations=[
            {
                "layer": layer,
                "component": "block_output",
                "low_rank_dimension": 4,
                "intervention": pyreft.LoreftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=4
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
        default=0,
        help="batch size; 0 = sequential (paper: 256)",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, choices=[128, 2048])
    args = parser.parse_args()

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
        inputs = tokenizer(
            load_examples(args.batch), return_tensors="pt", padding=True
        ).to(device)
        input_dict = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        reft_model.generate(input_dict, **warmup_kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _, generated = reft_model.generate(input_dict, **gen_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = generated_token_count(generated, inputs["input_ids"].shape[1])
        report(tokens, elapsed, args.batch)
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
