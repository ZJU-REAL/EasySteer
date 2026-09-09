# ReFT training (learning-based steering)

`easysteer.reft` includes a local implementation of
[pyreft](https://github.com/stanfordnlp/pyreft). It trains an intervention on a frozen
Hugging Face model, then exports a checkpoint that can be used by the inference
engine through a [steering payload](steering.md#steering-with-your-own-tensors).

For the analysis-based route without training, see
[Extracting steering vectors](extracting-vectors.md).

## Train a bias intervention

The shared `train_reft` helper handles model loading, the training data module,
trainer construction, and checkpoint saving. Run this script in the installed
EasySteer environment with a GPU:

```python
from easysteer.reft.train import EMOJI_EXAMPLES, train_reft

train_reft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    examples=EMOJI_EXAMPLES,
    intervention="bias",
    layer=8,
    save_dir="results/emoji_bias",
    output_dir="results/emoji_bias_training",
    num_train_epochs=100,
    per_device_train_batch_size=8,
    learning_rate=3e-3,
)
```

The examples are instruction/response pairs. This helper supervises the last prompt
position and uses Qwen's chat format by default; pass `prompt_template` when using
a different model. Training arguments such as the learning rate are illustrative,
so evaluate the trained intervention on held-out prompts.

## Apply the checkpoint

Run inference in a separate process after training exits, so the training model no
longer occupies GPU memory. `from_pyreft` preserves the checkpoint's layer index.
A bias checkpoint becomes a `DirectionVector`, so its inference algorithm is
`direct`:

```python
from easysteer.vectors import from_pyreft
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enable_steer_vector=True,
    steer_algorithms=["direct"],
)
spec = SteeringSpec(vectors=[VectorSpec(
    data=from_pyreft("results/emoji_bias"),
    algorithm="direct",
    scale=1.0,
    apply=ApplySpec(prompt_positions=[-1]),
)])
prompt = "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n"
outputs = llm.generate(
    prompt,
    SamplingParams(temperature=0.0, max_tokens=128),
    steering=spec,
)
print(outputs[0].outputs[0].text)
```

The selection matches the last prompt position used in training. Applying the
intervention to every generated token is a different experiment; choose that
explicitly with `generation="all"` if needed.

## LoReFT and lower-level APIs

For LoReFT, train with `intervention="loreft"` and `low_rank_dimension=4`, save to
a separate checkpoint directory, and change both `steer_algorithms` and the
`VectorSpec.algorithm` to `"loreft"`. The same `from_pyreft` adapter then produces
a `ReftIntervention` payload. The adapter accepts a single intervention checkpoint
(one weights file and one configuration file), rather than an arbitrary collection
of saved interventions.

The lower-level exports are under `easysteer.reft.pyreft`, including `ReftConfig`,
`get_reft_model`, `LoreftIntervention`, and `ReftTrainerForCausalLM`.
`BiasIntervention` lives in `easysteer.reft.pyreft.reft.algorithms`. For a complete
LoReFT experiment, see the [replication gallery](../replications/index.md).
