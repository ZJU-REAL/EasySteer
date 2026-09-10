# Math steering experiment

Evaluate SEAL on MATH500 and GSM8K with
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. Run notebooks from this directory
using the installed EasySteer environment, with `math-verify` installed.
`EASYSTEER_MODEL` can point to a local copy of the same model.

- `evaluate.ipynb` compares baseline and steering in one engine, with CUDA
  graphs enabled by default. It reports accuracy and mean generated tokens.
- `data_construction.ipynb` rebuilds the three reasoning vectors from 1,000
  training traces. Evaluation can reuse the committed `*_avg_vector.gguf` files.
- `common.py` shares the original reasoning prompt and answer scoring.

## Evaluation data

Download [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) and
[GSM8K](https://huggingface.co/datasets/openai/gsm8k) once:

```python
import json
from pathlib import Path

from datasets import load_dataset

datasets = {
    "math500": load_dataset("HuggingFaceH4/MATH-500", split="test"),
    "gsm8k_test": load_dataset("openai/gsm8k", "main", split="test"),
}
for name, rows in datasets.items():
    Path(f"{name}.json").write_text(json.dumps(list(rows), ensure_ascii=False))
```

`EASYSTEER_DATA_DIR` selects an existing data directory. The default evaluates
all 500 MATH questions and 1,319 GSM8K questions; `EASYSTEER_LIMIT` selects a
smaller initial subset for a quick check. Both conditions use greedy decoding
with an 8,192-token limit, and the explicit reasoning prefix from the experiment.
GSM8K scoring extracts the reference answer after `####`.

Steering adds `execution - reflection - transition` at layer 20, scale 0.5,
only on paragraph-break tokens during generation. The sum is built from the
three source vectors in memory, so there is no separate merged file to keep
up to date.

The notebook contains computed summaries and examples. Full generated answers
go to the ignored `.runtime/` directory; downloaded datasets are also ignored.

## Results

The full evaluation on 2026-09-10 (UTC) used vLLM 0.29.0, math-verify 0.9.0, and one NVIDIA RTX 6000D,
with the bundled vectors reused without retraining or extraction.

| Dataset | Questions | Baseline accuracy | SEAL accuracy | Baseline mean tokens | SEAL mean tokens |
|---|---:|---:|---:|---:|---:|
| MATH500 | 500 | 63.80% | 68.60% | 4,488.8 | 3,633.6 |
| GSM8K | 1,319 | 75.06% | 80.74% | 2,749.5 | 1,688.5 |

These are the executed notebook's results for the settings above. Its final
cell displays the first incorrect-to-correct example from each dataset.

## Rebuilding vectors

Provide `math_train_1000.json`, a JSON list of the experiment's 1,000 MATH
training problem strings, in the data directory. This historical subset is
not distributed here; selecting another subset produces a new extraction run.
The construction notebook generates traces, captures paragraph-break rows in
small batches, and accumulates category sums before exporting each vector.
Use the same model for construction and evaluation.
