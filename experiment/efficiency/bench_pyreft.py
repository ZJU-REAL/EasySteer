# SPDX-License-Identifier: Apache-2.0
"""pyreft (HF transformers) efficiency benchmark (EasySteer paper, 5.1).

The paper's framework comparison uses all-layer intervention with
zero-valued vectors, so a rank-4 LoReFT is built on every layer and its
parameters are zeroed — output text matches an unsteered run. Sequential
by default; --batch N (paper: 256) times one padded batch instead.
"""

import argparse
import os
import time

import torch
import transformers

import easysteer.reft.pyreft as pyreft
from common import MODEL, N_SEQUENTIAL, load_examples, report


def load_reft_model(device):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL, padding_side="left", use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # All-layer rank-4 LoReFT with zeroed parameters: the intervention
    # runs on every layer but is the identity, matching the paper's
    # zero-valued-vector setup.
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
        for intervention in reft_model.interventions.values():
            module = intervention[0] if isinstance(intervention, (list, tuple)) else intervention
            for p in module.parameters():
                p.zero_()
    reft_model.set_device(device)
    reft_model.eval()
    return reft_model, tokenizer


def generated_token_count(generated, attention_mask, pad_token_id):
    """Per-sample valid output length minus input length, summed."""
    input_lens = attention_mask.sum(dim=1).to(generated.device)
    out_valid = (generated != pad_token_id).sum(dim=1)
    return int((out_valid - input_lens).clamp(min=0).sum().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=0,
                        help="batch size; 0 = sequential (paper: 256)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        choices=[128, 2048])
    args = parser.parse_args()

    device = "cuda"
    reft_model, tokenizer = load_reft_model(device)
    gen_kwargs = dict(
        intervene_on_prompt=False,
        max_new_tokens=args.max_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
    )

    if args.batch:
        inputs = tokenizer(load_examples(args.batch), return_tensors="pt",
                           padding=True).to(device)
        input_dict = {"input_ids": inputs["input_ids"],
                      "attention_mask": inputs["attention_mask"]}
        start = time.time()
        _, generated = reft_model.generate(input_dict, **gen_kwargs)
        elapsed = time.time() - start
        tokens = generated_token_count(generated, inputs["attention_mask"],
                                       tokenizer.pad_token_id)
        report(tokens, elapsed, args.batch)
    else:
        examples = load_examples(N_SEQUENTIAL)
        prepared = [tokenizer(e, return_tensors="pt").to(device)
                    for e in examples]
        tokens = 0
        start = time.time()
        for inputs in prepared:
            input_dict = {"input_ids": inputs["input_ids"],
                          "attention_mask": inputs["attention_mask"]}
            _, generated = reft_model.generate(input_dict, **gen_kwargs)
            tokens += generated_token_count(
                generated, inputs["attention_mask"], tokenizer.pad_token_id)
        elapsed = time.time() - start
        report(tokens, elapsed, len(examples))


if __name__ == "__main__":
    main()
