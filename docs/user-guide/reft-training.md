# ReFT training (learning-based steering)

`easysteer.reft` includes a local implementation of
[pyreft](https://github.com/stanfordnlp/pyreft). It trains an intervention on a frozen
Hugging Face model, then exports a checkpoint that can be used by the inference
engine through a [steering payload](steering.md#steering-with-your-own-tensors).

For the analysis-based route without training, see
[Extracting steering vectors](extracting-vectors.md).

## Train a bias intervention

The shared `train_reft` helper handles model loading, the training data module,
trainer construction, and checkpoint saving. Run this script from the repository
root in the installed EasySteer environment with a GPU:

```python
from easysteer.reft.train import train_reft
from examples.reft import EMOJI_EXAMPLES

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

The helper saves its prompt template, position selection, and standard linear
activation setting in `config.json` under `easysteer_training`. A command-line
version is available in `examples/reft.py`.

## Apply the checkpoint

Run inference in a separate process after training exits, so the training model no
longer occupies GPU memory. `from_pyreft` preserves the checkpoint's layer index.
A bias checkpoint becomes a `DirectionVector`, so its inference algorithm is
`direct`:

```python
import json
from pathlib import Path

from easysteer.vectors import from_pyreft
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enable_steer_vector=True,
    steer_algorithms=["direct"],
)
config = json.loads(Path("results/emoji_bias/config.json").read_text())
training = config["easysteer_training"]
spec = SteeringSpec(vectors=[VectorSpec(
    data=from_pyreft("results/emoji_bias"),
    algorithm="direct",
    scale=1.0,
    apply=ApplySpec(**training["apply"]),
)])
prompt = training["prompt_template"] % "Who are you?"
outputs = llm.generate(
    prompt,
    SamplingParams(temperature=0.0, max_tokens=128),
    steering=spec,
)
print(outputs[0].outputs[0].text)
```

The prompt preserves the training helper's exact `prompt_template`, and the
selection matches the last prompt position used in training. Applying the
intervention to every generated token is a different experiment; choose that
explicitly with `generation="all"` if needed.

Older checkpoints may not contain `easysteer_training`. Supply their original
prompt format and `ApplySpec` explicitly; the bundled emoji checkpoints use
`ApplySpec(prompt_positions=[-1])`.

## LoReFT and lower-level APIs

For LoReFT, train with `intervention="loreft"` and `low_rank_dimension=4`, save to
a separate checkpoint directory, and change both `steer_algorithms` and the
`VectorSpec.algorithm` to `"loreft"`. The same `from_pyreft` adapter then produces
a `ReftIntervention` payload. The adapter accepts one explicitly identified
LoReFT or bias intervention on `block_output`, with `unit="pos"` and one unit.
Other component targets, intervention types, and multi-intervention checkpoints
raise an error before weights are loaded. An attention-output checkpoint cannot
be loaded as hidden-state LoReFT merely because its tensor width matches.

LoReFT conversion implements the standard linear activation used by the training
helper. Explicit nonlinear `act_fn` metadata is rejected. Older checkpoints
without activation metadata are interpreted as standard linear LoReFT; the
activation cannot be recovered from the saved weight tensors alone.

The lower-level exports are under `easysteer.reft.pyreft`, including `ReftConfig`,
`get_reft_model`, `LoreftIntervention`, and `ReftTrainerForCausalLM`.
`BiasIntervention` lives in `easysteer.reft.pyreft.reft.algorithms`. For a complete
LoReFT experiment, see the [replication gallery](../replications/index.md).

The training helper resolves the width of registered sequence components from the
model profile. Head selection and custom module paths require the lower-level
PyReFT API, which also provides multiple representations, per-example positions,
and `h.pos` units. Those training capabilities are broader than the checkpoint
adapter's supported inference targets; exporting a payload does not translate an
arbitrary PyReFT configuration into a vLLM intervention.
