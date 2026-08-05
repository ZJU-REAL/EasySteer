# Math steering experiment

Data files are not committed (datasets and inference artifacts stay out
of the repository):

- `math_train_1000.json`, `gsm8k_test.json`, `math500.json` — built by
  `data_construction.ipynb` from the public MATH / GSM8K / MATH-500
  datasets (use `HF_ENDPOINT=https://hf-mirror.com` on the lab servers).
- `results.json` — produced by running `baseline.ipynb` / `steer.ipynb`.

The steering vectors (`*_avg_vector.gguf`) are committed: they are the
experiment's extracted artifacts and are reused by the efficiency
benchmarks.
