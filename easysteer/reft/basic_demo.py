"""Train a LoReFT emoji-response intervention and chat with it.

Set REFT_MODEL_PATH (or edit the call below) to the base model directory.
"""

import os

from easysteer.reft.train import EMOJI_EXAMPLES, generate_reft, train_reft

reft_model, tokenizer = train_reft(
    model_path=os.environ.get("REFT_MODEL_PATH"),
    examples=EMOJI_EXAMPLES,
    intervention="loreft",
    num_train_epochs=100.0,
    save_dir="./results/loreft",
)
print(generate_reft(reft_model, tokenizer, "Who are you?"))
