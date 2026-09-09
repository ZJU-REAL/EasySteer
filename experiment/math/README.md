# Math steering experiment

Run notebooks with this directory as their working directory. The default model
is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`; `EASYSTEER_MODEL` can point to a
local copy. Use the same model when extracting and applying the saved vectors.

Data files are not committed (datasets and inference artifacts stay out
of the repository):

- `math_train_1000.json` — a JSON list of 1000 MATH training problem strings.
- `gsm8k_test.json` — a JSON list of GSM8K records with `question` and `answer`.
- `math500.json` — a JSON list of MATH-500 records with `problem` and `answer`.

Prepare these input datasets before running the notebooks.
`data_construction.ipynb` reads the MATH training file to extract the three
reasoning vectors; it does not download or create the input datasets.

The steering vectors (`*_avg_vector.gguf`) are committed: they are the
experiment's extracted artifacts and are reused by the efficiency
benchmarks.
