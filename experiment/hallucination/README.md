# Hallucination steering experiment

Run notebooks with this directory as their working directory and the installed
EasySteer environment as their kernel. `baseline.ipynb`, `data.ipynb` and
`steer.ipynb` default to `Qwen/Qwen2.5-1.5B-Instruct`; set `EASYSTEER_MODEL` to a
local copy of that model. The saved steering vectors must match the model.

`data.ipynb` uses `easysteer.capture.capture` and the labelled
`easysteer.extraction.extract` API. It captures only layers 14–27, which are
used by the steering notebook, and reuses each fold's capture for DiffMean,
exact centered PCA and linear probe. Capture retains the default 256 MiB
raw-data budget; exact extraction explicitly allows 1 GiB of estimated
working allocations. Captures are released after each fold. Incremental PCA
is not substituted for the experiment's exact centered PCA.

For two GPUs, set `CUDA_VISIBLE_DEVICES=0,1` and `EASYSTEER_TP=2` before
starting the notebook kernel. The default is one GPU. Capture supports
compiled execution and does not require eager mode.

`eval.ipynb` uses the separate `IAAR-Shanghai/xFinder-qwen1505` evaluator. Set
`EASYSTEER_EVAL_MODEL` to reuse a local copy of that evaluator. Its prompt template
is specific to xFinder-qwen1505.

The committed answers and GGUF vectors are historical experiment artifacts.
Notebook outputs have been cleared; updated code has not been rerun at full scale.
Record the model, package versions and generation settings when comparing scores.
