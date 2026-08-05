# Datasets for the replications

Public datasets and derived data files are not committed. The notebooks
expect them at the paths below; fetch them once before running:

```python
# MATH-500 (used by seal, fractreason, controlingthinkingspeed):
from datasets import load_dataset
import json
rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
problems = [r["problem"] for r in rows]
for d in ("seal", "fractreason", "controlingthinkingspeed"):
    json.dump(problems, open(f"{d}/math500.json", "w"))

# Alpaca (used by cast):
rows = load_dataset("tatsu-lab/alpaca", split="train")
json.dump([dict(r) for r in rows], open("cast/alpaca.json", "w"))
```

On the lab servers use the HF mirror (`HF_ENDPOINT=https://hf-mirror.com`).

`*_problems.json` subsets are derived inside the notebooks on first run.
Small curated task inputs (contrastive pairs, style examples, expert word
lists) remain committed — they are part of the replication itself.
