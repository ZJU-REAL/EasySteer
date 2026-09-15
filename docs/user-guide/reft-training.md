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

Each example is an instruction/response pair. By default, the transformation
applies at the last prompt token. Response tokens supply the causal
language-model loss, and prompt tokens are excluded from the loss. The loss is
averaged over response tokens across the whole effective batch, including all
distributed ranks and accumulated microbatches. Longer responses therefore
contribute more tokens. The base model stays frozen and in evaluation mode,
while the small steering transformation receives gradients.

The default prompt template is Qwen's chat format. Pass `prompt_template` for
other formats, or set `EASYSTEER_MODEL_PATH` and pass `model_path=None`.
Evaluate learned adapters on held-out prompts; these example hyperparameters
are illustrative. A command-line example is available in `examples/training.py`.

## Select training positions

Pass `apply` as vLLM's `ApplySpec`, `SelectSpec`, or the equivalent JSON-compatible
dictionary. Training calls the same token-selection resolver as `vllm-steer`:

```python
from vllm.steer_vectors import ApplySpec

model, tokenizer = train(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    examples=EMOJI_EXAMPLES,
    algorithm="loreft",
    layer=8,
    rank=4,
    apply=ApplySpec(
        prompt_window=(-3, None),
        exclude_prompt_positions=[-2],
        generation_window=(0, 4),
    ),
    save_dir="results/emoji_selected",
)
```

This selects the last three prompt tokens except the second-to-last token, plus
the first four response tokens. Prompt and generation phases are independent;
include selectors form a union, and exclusions subtract from that union.
Token IDs, positions, half-open windows, and `prompt="all"` /
`generation="all"` have the same meaning in training and inference. Prompt
positions count from each unpadded prompt: `-1` selects its last token, and a
positive index past the end clamps to that token. Generation positions start at
zero and cannot be negative.

Selection identifies the hidden state being transformed, not the token being
predicted. The last prompt state predicts the first response token.
`generation_positions=[0]` transforms the first response token's state when
predicting the second response token. During training, response token IDs come
from the reference answer; during generation, they come from the model's output.
The same selection rule applies in both cases, including with the KV cache.
Padding is never selected.

## Train on multiple GPUs

Use PyTorch DDP through `torchrun`, with one process per GPU. Device selection
and adapter seeding happen before model construction; each rank receives its
own complete frozen model and a synchronized adapter. For the example script:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  examples/training.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --algorithm loreft --epochs 200 --batch-size 5 \
  --save-dir results/emoji_loreft
```

Here the effective batch has 10 examples: 5 per GPU × 2 GPUs. With gradient
accumulation, multiply again by `gradient_accumulation_steps`. Keep this global
batch size and the training seed fixed when comparing runs. DDP synchronizes
adapter gradients and only rank zero writes adapter checkpoints. Distributed
floating-point reduction can introduce small numerical differences.

Compare the interval `loss` values from `logging_steps`: Trainer gathers these
across ranks. Its final `train_loss` summary can include a rank-local remainder
when the last step is not logged, including with `logging_strategy="no"`. That
summary does not affect the globally normalized gradients or parameter updates.

DDP improves throughput; each GPU must still fit the full base model and its
activations. Training does not support model sharding, tensor parallelism,
FSDP, DeepSpeed, or `DataParallel`. Multiple visible GPUs without `torchrun` are
rejected. For one GPU, set `CUDA_VISIBLE_DEVICES=0` and launch with `python`.
Keep `device="cuda"` under `torchrun` so the launcher assigns each rank's device.
CPU DDP is available with `device="cpu"`.

Gradient checkpointing is unsupported because it replays the temporary steering
hooks during backward. Native checkpoints store the materialized adapter, so
exact optimizer-state resume through the Trainer is also unsupported. Load a
checkpoint for inference with the APIs below.

## Apply the checkpoint

A training run saves `steering_adapter.json`: a versioned configuration and native
payload, including its algorithm, component, layer, prompt template and token
selection. It contains no Python class paths or frozen base-model weights.
New checkpoints use format version 2. The loader also accepts version 1
checkpoints with their original last-prompt-token selection.

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

`to_spec()` preserves the complete training selection. Its default remains
`prompt_positions=[-1]`; explicitly changing that selection changes the experiment.
For payload-only integrations, `easysteer.vectors.from_training(path)` and
`vectors.load(path, format="training")` return the native payload; the caller
then supplies the remaining spec fields.

`load(model_path, save_dir)` reloads the adapter on a Hugging Face model, and
`generate(model, tokenizer, instruction)` uses its stored prompt template and
selection for both prompt processing and subsequent generation. It supports
the default dynamic KV cache and full-sequence recomputation with
`use_cache=False`. Hugging Face static caches are explicitly unsupported.

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
