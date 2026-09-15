# Training steering adapters

`easysteer.training` trains a small transformation on a frozen Hugging Face causal
language model and exports a native vLLM steering payload. It uses PyTorch and the
Transformers Trainer directly; PyReFT and PyVene are no longer bundled or required.

Install the training dependencies in your EasySteer environment:

```bash
pip install -e '.[training]'
```

## Train

```python
from easysteer.training import train
from examples.training import EMOJI_EXAMPLES

model, tokenizer = train(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    examples=EMOJI_EXAMPLES,
    algorithm="direct",
    component="hidden_states",
    layer=8,
    save_dir="results/emoji_direct",
    output_dir="results/emoji_direct_training",
    num_train_epochs=100,
    per_device_train_batch_size=8,
    learning_rate=3e-3,
)
```

`direct` learns an additive direction. `loreft` learns the standard linear LoReFT
transformation; select it with `algorithm="loreft", rank=4`. These names match
vLLM's inference algorithms. Both currently support `component="hidden_states"`,
at the output of a supported decoder layer. Attention and router-logit training
are not implemented. Unsupported components fail before the model is loaded.

Each example is an instruction/response pair. The transformation applies at the
last prompt token; the response tokens supply the causal language-model loss.
Prompt tokens are excluded from the loss. The base model stays frozen and in
evaluation mode, while the small steering transformation receives gradients.
Gradient checkpointing is not supported by this training wrapper.

The default prompt template is Qwen's chat format. Pass `prompt_template` for
other formats, or set `EASYSTEER_MODEL_PATH` and pass `model_path=None`.
Evaluate learned adapters on held-out prompts; these example hyperparameters
are illustrative. A command-line example is available in `examples/training.py`.

## Apply the checkpoint

A training run saves `steering_adapter.json`: a versioned configuration and native
payload, including its algorithm, component, layer, prompt template and token
selection. It contains no Python class paths or frozen base-model weights.

Run inference after the training process exits to release its GPU memory:

```python
from easysteer.training import load_checkpoint
from vllm import LLM, SamplingParams

checkpoint = load_checkpoint("results/emoji_direct")
spec = checkpoint.to_spec()
llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enable_steer_vector=True,
    steer_algorithms=[checkpoint.config.algorithm],
)
prompt = checkpoint.config.prompt_template % "Who are you?"
outputs = llm.generate(
    prompt,
    SamplingParams(temperature=0.0, max_tokens=128),
    steering=spec,
)
print(outputs[0].outputs[0].text)
```

`to_spec()` preserves the training selection (`prompt_positions=[-1]`). Applying
an adapter to generated tokens or every prompt token changes the experiment.
For payload-only integrations, `easysteer.vectors.from_training(path)` and
`vectors.load(path, format="training")` return the native payload; the caller
then supplies the remaining spec fields.

`load(model_path, save_dir)` reloads the adapter on a Hugging Face model, and
`generate(model, tokenizer, instruction)` uses its stored prompt template and
last-prompt-token selection. It does not steer subsequent cached decode steps.

## Migration from the removed ReFT API

The `easysteer.reft` package and its lower-level PyReFT/PyVene APIs have been
removed. Update training calls as follows:

| Former API | Native API |
|---|---|
| `from easysteer.reft.train import train_reft` | `from easysteer.training import train` |
| `intervention="bias"` | `algorithm="direct"` |
| `intervention="loreft"` | `algorithm="loreft"` |
| `component="block_output"` | `component="hidden_states"` |
| `low_rank_dimension=4` | `rank=4` |
| `REFT_MODEL_PATH` | `EASYSTEER_MODEL_PATH` |

The native wrapper does not implement PyVene's source/base interventions,
model profiles, arbitrary component paths, classification trainers, or multiple
representations. Its supported targets and exported payloads match the inference
contract.

Existing published checkpoints remain usable through
`easysteer.vectors.from_pyreft`. This file adapter does not import PyReFT: it
validates known metadata and reads tensor weights. It accepts one bias or linear
LoReFT checkpoint on `block_output` with `unit="pos"` and one unit. Unsupported
targets, nonlinear activations and multi-intervention checkpoints are rejected.
For historical files, supply their original prompt format and `ApplySpec`
explicitly; the bundled emoji checkpoint uses `prompt_positions=[-1]`.

The mathematical LoReFT algorithm and native `ReftIntervention` payload name
remain compatible. They do not imply a dependency on the former framework.
