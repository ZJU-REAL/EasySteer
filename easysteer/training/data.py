# SPDX-License-Identifier: Apache-2.0
"""Response-only supervision with explicit prompt lengths for token selection."""

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class SupervisedDataset(Dataset):
    def __init__(self, tokenizer, examples, prompt_template, max_length=2048):
        if type(max_length) is not int or max_length < 2:
            raise ValueError("max_length must be an integer of at least two tokens")
        if not examples:
            raise ValueError("training examples must not be empty")
        self.rows = []
        for index, example in enumerate(examples):
            if (
                not isinstance(example, (list, tuple))
                or len(example) != 2
                or not all(isinstance(text, str) and text for text in example)
            ):
                raise ValueError(
                    f"example {index} must contain nonempty prompt/response strings"
                )
            prompt = tokenizer(prompt_template % example[0])["input_ids"]
            response = tokenizer(example[1], add_special_tokens=False)["input_ids"]
            if not prompt or not response:
                raise ValueError(
                    f"example {index} has an empty prompt or response token sequence"
                )
            if len(prompt) >= max_length:
                raise ValueError(
                    f"example {index} leaves no room for response supervision"
                )
            if (
                tokenizer.eos_token_id is not None
                and response[-1] != tokenizer.eos_token_id
            ):
                response = [*response, tokenizer.eos_token_id]
            # Separate tokenization preserves the same prompt prefix that
            # inference receives, including at a BPE merge boundary.
            ids = [*prompt, *response][:max_length]
            self.rows.append(
                {
                    "input_ids": ids,
                    "labels": [-100] * len(prompt) + ids[len(prompt) :],
                    "prompt_lengths": len(prompt),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class SupervisedCollator:
    def __init__(self, pad_token_id: int):
        if pad_token_id is None:
            raise ValueError("training requires a padding token")
        self.pad_token_id = pad_token_id

    def __call__(self, rows: Sequence[dict]) -> dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("training batch must not be empty")
        length = max(len(row["input_ids"]) for row in rows)
        shape = (len(rows), length)
        batch = {
            "input_ids": torch.full(shape, self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros(shape, dtype=torch.long),
            "labels": torch.full(shape, -100, dtype=torch.long),
            "prompt_lengths": torch.tensor(
                [row["prompt_lengths"] for row in rows], dtype=torch.long
            ),
        }
        for index, row in enumerate(rows):
            size = len(row["input_ids"])
            batch["input_ids"][index, :size] = torch.tensor(row["input_ids"])
            batch["attention_mask"][index, :size] = 1
            batch["labels"][index, :size] = torch.tensor(row["labels"])
        return batch
