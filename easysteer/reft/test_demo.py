"""Load the BiasIntervention model saved by ssv.py and chat with it.

Set REFT_MODEL_PATH (or edit the call below) to the base model directory.
"""

import os

from easysteer.reft.train import generate_reft, load_reft

reft_model, tokenizer = load_reft(
    os.environ.get("REFT_MODEL_PATH"), "./results/ssv"
)
print(generate_reft(reft_model, tokenizer, "Who are you?"))
