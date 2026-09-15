# Extracting steering vectors

Two complementary routes turn captured hidden states into interventions.

## Analysis-based: `easysteer.extraction`

Derive directions from captured activations while keeping model weights fixed.
`extract` accepts a `CaptureResult` or a one-pass iterator of capture batches,
with one Boolean or integer 0/1 label per sample. Layer IDs and component names
come from the capture, so attention outputs remain attention directions.

```python
from easysteer.capture import capture_batches
from easysteer.extraction import extract
from vllm.capture import SelectSpec
from vllm.steer_vectors import ApplySpec

# prompts and labels have the same order; labels can also be an iterator.
batches = capture_batches(
    llm, prompts, layers=[10, 11, 12],
    select=SelectSpec(prompt_positions=[-1]),
    budget_bytes=256 * 1024**2,
)
vector = extract(batches, labels, method="diffmean")
spec = vector.to_spec(apply=ApplySpec(generation="all"), scale=1.5)
outputs = llm.generate(new_prompts, steering=spec)
```

DiffMean releases each consumed batch and retains two running class sums per
layer. No accumulator loop or conversion to nested lists is needed. Label
counts, missing rows, changed layers and inconsistent capture provenance produce
explicit errors. Labels follow batch/sample order; `sample_indices` remains
available on each capture for tracking the original dataset. It is not used
to index or reorder the label iterable.

`token_pos=-1` selects the last **captured** row of each sample. Other choices
are an integer row index, `"first"`, `"last"`, `"mean"`, `"max"` or `"min"`.
The last two select the row with the largest or smallest L2 norm. Mean pooling
uses bounded token chunks, including when continuous batching interleaves one
sample's rows. Each sample contributes one pooled row, so long responses do not
receive extra weight.

### Working memory and PCA

The default `max_working_bytes=256 * 1024**2` preflights extraction's estimated
numerical allocations. It excludes input captures, Python objects and numerical
library/runtime overhead; it is not an operating-system RSS limit. Capture has
its own host/device budgets. To process a large dataset, consume
`capture_batches(...)` directly or iterate over saved captures; turning the
iterator into a list retains all input activations.

| Method | Accepted input | Numerical storage |
|---|---|---|
| `diffmean` | Single capture or batch iterator | Class sums proportional to total captured layer width |
| `incremental_pca` | Single capture or batch iterator | Bounded per-layer row buffers plus one SVD's scratch space |
| `pca`, `lat`, `linear_probe`, `iti` | Single materialized or memory-mapped capture | One layer's reduced dataset and estimator scratch, checked before extraction |

Incremental PCA is an explicit approximation, fit on positive examples. Negative
examples orient its sign when present. Its result can depend on sample order and
`pca_batch_size`; ordinary `method="pca"` never switches to it automatically.
For a single finite capture, the allocated PCA buffer is capped by its sample
count. Requested batch size remains recorded in the metadata.

```python
vector = extract(
    batches, labels, method="incremental_pca", pca_batch_size=32,
    max_working_bytes=128 * 1024**2,
)

# For a dataset that fits the materialized-method working budget:
vector = extract(capture_result, labels, method="pca", variant="center")
probe = extract(capture_result, labels, method="linear_probe", C=1.0)
```

An iterator is consumed by one extraction call. To compare algorithms, create a
fresh iterator over saved batches instead of rerunning model capture.
See the [saved-dataset example](hidden-state-capture.md#process-a-dataset-in-batches)
for reopening batches in order. Exact
covariance remains available through
`MomentsAccumulator(track_second_moment=True, max_working_bytes=...)`, but it
requires quadratic float64 storage in layer width. Its budget check happens
before statistics change and also reserves covariance/eigendecomposition scratch.
Use incremental PCA when those matrices do not fit.

### Export and existing interfaces

`to_spec` requires an explicit inference `ApplySpec`; capture positions are
analysis provenance and are never reused as an implicit steering policy. It
selects `direct` for hidden states and `attention_add` for attention heads.
Statistical router-logit directions can be analyzed or exported, but cannot be
converted to a `moe_router` intervention: that algorithm requires explicit
expert selections and modes.

```python
from easysteer.extraction import StatisticalControlVector

vector.export_gguf("vectors/diffmean.gguf")
restored = StatisticalControlVector.import_gguf("vectors/diffmean.gguf")
spec = restored.to_spec(apply=ApplySpec(generation="all"))
```

GGUF preserves directions, component, model hint and nested extraction metadata,
including integer layer keys, strings, Booleans and token pooling settings.
Vector metadata records the global capture selection and whether any per-prompt
overrides were used. The saved capture retains each override and the exact row
positions; vectors do not retain a corpus-sized list of sample selections.
Older GGUF files remain readable. When their component is unknown, supply
`component="hidden_states"` or `component="attention_heads"` explicitly to
`to_spec`. A conflicting known component is rejected.

The existing functions (`extract_diffmean_control_vector`,
`extract_pca_control_vector`, `extract_lat_control_vector`,
`extract_linear_probe_control_vector`, `extract_statistical_control_vector`) and
extractor classes remain available for index-based analysis. They accept
`positive_indices` and optional `negative_indices`; omitted negatives are the
complement. These lower-level interfaces require the caller to size the input
and working storage. The generic index-based selector is `algorithm="pca"`,
with `variant="standard"`, `"diff"` or `"center"`; historical `method` aliases
remain supported. Use `CaptureResult` to preserve true layer IDs. Legacy nested
`[sample][layer][token]` inputs infer layer IDs from positions and have no target
provenance.

Extraction normalizes directions by default, except ITI, which preserves its
projection scale. This differs from inference-time `VectorSpec.normalize=True`,
which rescales the transformed hidden state to its original norm. Standardized
linear probes export coefficients converted back to raw activation coordinates.
SAE helpers locate and export interpretable decoder directions; see the
[API reference](../api-reference/steer.md).

### ITI attention head directions

ITI takes training and validation captures from
[`stream="attention_heads"`](hidden-state-capture.md#attention-head-outputs).
Split data by question before capture so different answers to one question
cannot cross the training, validation, and test boundaries.

```python
from pathlib import Path

from easysteer.extraction import extract
from vllm.steer_vectors import ApplySpec

iti = extract(
    train_capture, train_labels, method="iti",
    validation_hidden_states=validation_capture,
    validation_labels=validation_labels,
    # Query head counts come from the capture layouts.
    top_k=1,
)
spec = iti.to_spec(apply=ApplySpec(generation="all"), scale=15.0)
Path(".runtime/iti").mkdir(parents=True, exist_ok=True)
iti.export_gguf(".runtime/iti/iti.gguf")
```

Both label iterables contain one Boolean or integer 0/1 per sample. The
index-based `ITIExtractor.extract` also remains available: unlisted answer
indices are negatives unless explicit negative indices are provided. Probes fit
the training set and rank heads by validation accuracy.
Following [ITI](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html),
the selected head directions use the truthful-minus-false means from both
development splits, normalized and multiplied by their projection standard
deviation. Unselected head slices remain zero. The returned vector preserves
this scale; apply it with `algorithm="attention_add"`, `normalize=False`, and
`scale` set to the desired intervention strength. The bounded interface checks
training and validation layer widths before reading feature rows.

The [ITI notebook](https://github.com/ZJU-REAL/EasySteer/blob/main/replications/iti/iti.ipynb)
demonstrates Llama-2-7B-Chat with a supplied direction: 48 heads, strength 15,
one fixed TruthfulQA evaluation set, an accuracy comparison, and generated
examples. It uses the authors' QA prompt and loads the model once. The supplied
direction uses the official code's separate, unlabeled GEN calibration bank;
the extractor above uses its training and validation captures for that scale.

## Learning-based: `easysteer.training`

Train a native `direct` or `loreft` adapter on a frozen Hugging Face model with
`easysteer.training.train`. The checkpoint exports the payload and its target
through `load_checkpoint(path).to_spec()`. Training and inference use the same
algorithm and component names.

See [Training steering adapters](reft-training.md) for training, export and
migration from the removed ReFT API, or the [LoReFT replication](../replications/index.md)
for a complete notebook.
