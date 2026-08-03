# ReFT training (learning-based steering)

`easysteer.reft` reimplements [pyreft](https://github.com/stanfordnlp/pyreft):
it trains a parameterized intervention (e.g. SAV, LM-Steer, LoReFT, or a simple
`BiasIntervention`) on a **frozen** HuggingFace model with a standard `transformers`
trainer, then saves the learned representation so it can be applied at inference time
with a [`SteeringSpec`](steering.md).

For the analysis-based (no-training) route, see
[Extracting steering vectors](extracting-vectors.md).

## End-to-end example

Train a bias intervention that makes a model answer in emoji style:

```python
import torch
import transformers
import easysteer.reft as reft

# Load the base language model (weights stay frozen)
model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map="cuda"
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Configure ReFT: which layer/component to intervene on, and with what
reft_config = reft.ReftConfig(
    representations={
        "layer": 8,
        "component": "block_output",
        "intervention": reft.BiasIntervention(
            embed_dim=model.config.hidden_size
        ),
    }
)
reft_model = reft.get_reft_model(model, reft_config)

# Training data: prompts and target outputs
prompt_template = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
training_examples = [
    ["Who are you?", "🤖💬🌐🧠"],
    ["What's 2+2?", "🔢➕🔢➡️4️⃣"],
    ["Why is the sky blue?", "🌍🛡️☀️➡️🔵🌌"],
    # ... more training examples
]

data_module = reft.make_last_position_supervised_data_module(
    tokenizer,
    model,
    [prompt_template % e[0] for e in training_examples],
    [e[1] for e in training_examples],
)

training_args = transformers.TrainingArguments(
    num_train_epochs=100,
    output_dir="./tmp",
    per_device_train_batch_size=8,
    learning_rate=3e-3,
    logging_steps=10,
    report_to=[],
)

trainer = reft.ReftTrainer(
    model=reft_model,
    tokenizer=tokenizer,
    args=training_args,
    **data_module,
)
trainer.train()

# Save the trained intervention representation
reft_model.save("results/emoji_style")
```

## Applying the trained intervention

The saved representation is loaded at inference through the steering engine — e.g. with
`VectorSpec(source="results/emoji_style", algorithm="loreft", ...)` for LoReFT-format
files. See the [Steering guide](steering.md) for the spec language and the
[LoReFT replication](../replications/index.md) for a complete train-then-steer
notebook.

<!-- TODO: document the full easysteer.reft API surface (interventions, data modules,
trainers) and the on-disk format of saved representations; add an api-reference page
once docstring coverage is in place. -->
