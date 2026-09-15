# Replications

Run each notebook with its own directory as the working directory, using the
installed EasySteer package and the matching `vllm-steer` runtime. Model and
dataset access requirements are unchanged by the API migration.

`EASYSTEER_MODEL` selects a local copy of the notebook's base model.
`EASYSTEER_TP` selects inference tensor parallelism and defaults to one. For
example, start Jupyter with two available GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 EASYSTEER_TP=2 jupyter lab
```

Capture and steering use the same engine; eager execution is not required.
Choose a TP size supported by the model. The combined LoReFT notebook requires
one visible GPU for training and inference; its
[command-line recipe](loreft/README.md) supports DDP training and separate TP
inference. `EASYSTEER_TP` applies to the other notebooks' inference engines.

## Current APIs

- `easysteer.capture.capture` returns a labeled `CaptureResult` for a small
  dataset or an exact estimator. `capture_batches` yields bounded batches for
  streaming extraction and custom accumulators.
- `easysteer.extraction.extract(captures, labels, method=...)` takes one Boolean
  label per sample in input order. `True` selects the positive group and `False`
  the negative group. DiffMean notebooks consume capture batches directly.
- The PCA replications retain exact pair-centered PCA with `method="pca"` and
  `variant="center"`. They use a single capture rather than substituting
  incremental PCA, which would change the experiment. Fractional Reasoning and
  Controlling Thinking Speed allow 512 MiB of estimated extraction working
  allocations to accommodate all 500 problem pairs with their default models.
- `SelectSpec` controls which token rows are captured. Extraction's `token_pos`
  selects or pools those captured rows; `ApplySpec` independently controls
  inference positions. CAST pools its four selected prefix rows with
  `token_pos="mean"`; refusal-direction extraction keeps four separate positions.
- Raw `result.rows(layer)` are in fetch order. Use `result.token(sample, layer)`
  or `result.sample_rows(sample, layer)` for sample attribution. In batch loops,
  `result.sample_indices` maps local samples back to the original dataset.

Capture defaults bound retained activation data and device capture buffers;
extraction also checks its working allocations. These budgets exclude model
weights, KV cache and runtime overhead. Pass a new `storage_dir` to capture when
activations should persist to disk. See the [capture guide](../docs/user-guide/hidden-state-capture.md)
and [extraction guide](../docs/user-guide/extracting-vectors.md) for the controls.

SEAL uses token-weighted category averages, so it maintains category sums over
capture batches instead of binary-label extraction. SteerMoE captures
`router_logits` and constructs explicit expert-deactivation configurations;
an additive statistical direction is not a `moe_router` payload.

Native LoReFT training uses `easysteer.training`. The bundled historical
checkpoint and the `easysteer.vectors.from_pyreft` file adapter remain valid;
they do not require or import the removed PyReFT training framework.

## Inputs and results

The CAST construction expects a local `alpaca.json` list containing
`instruction` fields. Fractional Reasoning expects `math500_problems.json` as a
list of problem strings. Controlling Thinking Speed expects that filename to
contain records with `problem` and `answer` fields. Both reasoning notebooks
and SEAL evaluate from `math500.json`, a list of records with `problem` fields.
These dataset files are not bundled; prepare them in the corresponding notebook
directory before running construction or evaluation.

Saved notebook outputs and bundled vector/checkpoint files are preserved as
historical experiment records. Each notebook links the source snapshot of its
restored outputs. Output-cell metadata also records the original execution
count and source hash; current execution counters are empty because the updated
code has not been rerun. LoReFT's restored training logs come from the former
PyReFT implementation; its separate native-training validation is documented in
[the LoReFT guide](loreft/README.md).

API migrations should preserve these records and their provenance. Rerun all
cells and update the provenance note to record measurements for a new runtime.
API checks alone do not establish that a rerun matches the paper or saved results.
