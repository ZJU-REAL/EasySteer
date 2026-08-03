# SPDX-License-Identifier: Apache-2.0
"""repeng (HF transformers) efficiency benchmark.

Wraps the model in repeng's ControlModel with the SEAL execution vector
applied at strength 0 on layers 1-27 (the paper's all-layer, zero-valued
setup). Sequential by default; --batch N (paper: 64) times one padded
batch instead.
"""

import argparse
import time

import numpy as np

# repeng still uses the numpy<2 np.float_ alias; restore it before
# import (removed in NumPy 2.0).
np.float_ = np.float64

import torch
from repeng import ControlModel, ControlVector
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL, N_SEQUENTIAL, SEAL_VECTOR, load_examples, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=0,
                        help="batch size; 0 = sequential (paper: 64)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        choices=[128, 2048])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(MODEL).to("cuda")
    model = ControlModel(model, list(range(1, 28)))
    model.set_control(ControlVector.import_gguf(SEAL_VECTOR), 0)
    settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "max_new_tokens": args.max_tokens,
        "early_stopping": True,
    }

    if args.batch:
        inputs = tokenizer(load_examples(args.batch), return_tensors="pt",
                           padding=True).to(model.device)
        start = time.time()
        outputs = model.generate(**inputs, **settings)
        elapsed = time.time() - start
        with torch.no_grad():
            input_lens = inputs["attention_mask"].sum(dim=1)
            non_pad = (outputs != settings["pad_token_id"]).long().sum(dim=1)
            tokens = int((non_pad - input_lens).clamp(min=0).sum().item())
        report(tokens, elapsed, args.batch)
    else:
        prepared = [tokenizer(e, return_tensors="pt").to(model.device)
                    for e in load_examples(N_SEQUENTIAL)]
        tokens = 0
        start = time.time()
        for inputs in prepared:
            output = model.generate(**inputs, **settings)
            tokens += len(output.squeeze()) - inputs["input_ids"].shape[1]
        elapsed = time.time() - start
        report(tokens, elapsed, len(prepared))


if __name__ == "__main__":
    main()
