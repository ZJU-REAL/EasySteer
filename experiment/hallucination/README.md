# Hallucination steering experiment

Run notebooks with this directory as their working directory and the installed
EasySteer environment as their kernel. `baseline.ipynb`, `data.ipynb` and
`steer.ipynb` default to `Qwen/Qwen2.5-1.5B-Instruct`; set `EASYSTEER_MODEL` to a
local copy of that model. The saved steering vectors must match the model.

`eval.ipynb` uses the separate `IAAR-Shanghai/xFinder-qwen1505` evaluator. Set
`EASYSTEER_EVAL_MODEL` to reuse a local copy of that evaluator. Its prompt template
is specific to xFinder-qwen1505.

The committed answers and notebook outputs are historical experiment artifacts.
Record the model, package versions and generation settings when comparing scores.
