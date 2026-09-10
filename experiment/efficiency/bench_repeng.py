# SPDX-License-Identifier: Apache-2.0
"""repeng (HF transformers) efficiency benchmark.

Wraps the model in repeng's ControlModel with the SEAL execution vector
applied at strength 0 on layers 1-27 (the paper's all-layer, zero-valued
setup). Sequential by default; --batch N (paper: 64) times one padded
batch instead.
"""

import argparse
import time

from common import (
    MODEL,
    N_SEQUENTIAL,
    SEAL_VECTOR,
    load_examples,
    nonnegative_int,
    report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        type=nonnegative_int,
        default=0,
        help="batch size; 0 = sequential (paper: 64)",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, choices=[128, 2048])
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Retain the removed NumPy alias for the older repeng dependency.
    np.float_ = np.float64
    from repeng import ControlModel, ControlVector

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model = ControlModel(model, list(range(1, 28)))
    model.set_control(ControlVector.import_gguf(SEAL_VECTOR), 0)
    model.eval()
    settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "max_new_tokens": args.max_tokens,
        "min_new_tokens": args.max_tokens,
    }
    warmup_settings = {**settings, "min_new_tokens": 8, "max_new_tokens": 8}

    if args.batch:
        inputs = tokenizer(
            load_examples(args.batch), return_tensors="pt", padding=True
        ).to(model.device)
        model.generate(**inputs, **warmup_settings)
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model.generate(**inputs, **settings)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        # min_new_tokens == max_new_tokens keeps every generated suffix full;
        # subtract the padded prompt width, not the number of non-EOS input IDs.
        tokens = outputs.shape[0] * (outputs.shape[1] - inputs["input_ids"].shape[1])
        report(tokens, elapsed, args.batch)
    else:
        prepared = [
            tokenizer(e, return_tensors="pt").to(model.device)
            for e in load_examples(N_SEQUENTIAL)
        ]
        tokens = 0
        model.generate(**prepared[0], **warmup_settings)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for inputs in prepared:
            output = model.generate(**inputs, **settings)
            tokens += len(output.squeeze()) - inputs["input_ids"].shape[1]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        report(tokens, elapsed, len(prepared))


if __name__ == "__main__":
    main()
