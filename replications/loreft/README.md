# LoReFT emoji replication

This example reproduces the ten-example emoji demonstration associated with
[ReFT: Representation Finetuning for Language Models](https://arxiv.org/abs/2404.03592)
using native `easysteer.training` and vLLM inference. It is an instruction
memorization experiment, not a reproduction of the paper's benchmark tables.
PyReFT and PyVene are not required.

Install EasySteer with `pip install -e '.[training]'` and its matching
`vllm-steer` runtime. Run the following commands from the repository root.
Use the same base-model revision for training and evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 python replications/loreft/run.py train \
  --model Qwen/Qwen2.5-1.5B-Instruct --output-dir results/loreft-single

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  replications/loreft/run.py train \
  --model Qwen/Qwen2.5-1.5B-Instruct --output-dir results/loreft-ddp

CUDA_VISIBLE_DEVICES=0,1 python replications/loreft/run.py evaluate \
  --model Qwen/Qwen2.5-1.5B-Instruct --tp 2 \
  --checkpoint results/loreft-single \
  --compare-checkpoint results/loreft-ddp \
  --legacy-checkpoint replications/loreft/weight \
  --output-dir results/loreft-evaluation
```

The default recipe uses layer 8, `hidden_states`, rank 4, learning rate 0.004,
200 epochs, seed 42, and a **global** batch of ten examples. With two DDP ranks,
each GPU processes five examples. Each GPU holds a complete frozen base model;
DDP synchronizes the trainable adapter's gradients. Training TP, FSDP and
DeepSpeed are not supported.

Steering selects the last prompt token. Response tokens, including EOS, supply
the training loss. The checkpoint preserves this `ApplySpec` and the Qwen prompt
template; changing the selection at inference changes the experiment.
`training.json` records the recipe and loss history. `evaluation.json` records
all ten greedy completions, expected answers, token IDs and exact-match counts
for the baseline and each checkpoint. Evaluation fails if either of the two
answers demonstrated in the notebook is not reproduced by a native checkpoint.

The 200-epoch recipe was validated with Qwen2.5-1.5B-Instruct, PyTorch 2.13,
Transformers 5.16.1, and compiled TP=2 vLLM inference:

| Checkpoint | Exact answers |
| --- | --- |
| Unsteered base model | 0 / 10 |
| Native, one GPU | 9 / 10 |
| Native, two DDP GPUs | 9 / 10 |
| Bundled historical LoReFT | 9 / 10 |

Both native runs reproduced the notebook's two demonstrated answers exactly.
All three adapters missed the long road-trip answer; its completion differed
between checkpoints. These measurements demonstrate parity on this small
experiment, not perfect memorization or held-out generalization.

The native LoReFT transformation has the same inference formula as the
historical checkpoint. Its compact QR training parameterization differs from
the old framework's orthogonal parameterization, so training trajectories and
weights need not match. BF16 reductions can also produce different trajectories
between single-GPU and DDP training; compare task outputs and use the numerical
DDP tests for update correctness.

[`loreft.ipynb`](loreft.ipynb) provides an interactive single-GPU walkthrough.
The bundled `weight/` files remain historical checkpoint data and are read by
the standalone `easysteer.vectors.from_pyreft` adapter.
