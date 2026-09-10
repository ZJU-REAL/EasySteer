# Extracting steering vectors

Two complementary routes turn captured hidden states into interventions.

## Analysis-based: `easysteer.steer`

Derives directions from captured activations while keeping model weights fixed.
Available extractors: DiffMean, PCA, LAT, linear probe, ITI attention head
directions, and SAE feature vectors.

```python
from easysteer.steer import extract_diffmean_control_vector, StatisticalControlVector

control_vector = extract_diffmean_control_vector(
    all_hidden_states=capture_result,  # CaptureResult from hs.capture(...)
    positive_indices=[0, 1, 2, 3],
    negative_indices=[4, 5, 6, 7],
    token_pos=-1,       # which token's activation to use
    normalize=True,
)

control_vector.export_gguf("vectors/diffmean.gguf")
# ... later
control_vector = StatisticalControlVector.import_gguf("vectors/diffmean.gguf")
```

The exported GGUF file is what `VectorSpec(source=...)` consumes at inference time.
Pass `CaptureResult` directly to preserve true layer IDs. Legacy nested
`[sample][layer][token]` inputs are also accepted, but their layer keys are inferred
from list positions starting at zero. The selected rows must include every sample
needed by `positive_indices` and `negative_indices`.

Extractor `normalize=True` normalizes the extracted direction. This differs from
inference-time `VectorSpec.normalize=True`, which rescales the transformed hidden
state to its original norm.

Sibling functions follow the same shape: `extract_pca_control_vector`,
`extract_lat_control_vector`, `extract_linear_probe_control_vector`, and the generic
`extract_statistical_control_vector`. SAE helpers (`search_sae_features`,
`get_sae_feature_explanation`, `extract_sae_decoder_vector`) locate and export
interpretable SAE decoder directions. See the
[API reference](../api-reference/steer.md).

### ITI attention head directions

`ITIExtractor` takes training and validation captures from
[`stream="attention_heads"`](hidden-state-capture.md#attention-head-outputs).
Split data by question before capture so different answers to one question
cannot cross the training, validation, and test boundaries.

```python
from pathlib import Path

from easysteer.steer import ITIExtractor
from easysteer.vectors import from_control_vector

iti = ITIExtractor.extract(
    train_capture,
    positive_indices=train_true_answer_indices,
    validation_hidden_states=validation_capture,
    validation_positive_indices=validation_true_answer_indices,
    num_heads={
        layer: layout["num_heads"]
        for layer, layout in train_capture.layouts.items()
    },
    top_k=1,
)
payload = from_control_vector(iti)  # VectorSpec(data=payload, algorithm="attention_add", ...)
Path(".runtime/iti").mkdir(parents=True, exist_ok=True)
iti.export_gguf(".runtime/iti/iti.gguf")
```

Unlisted answer indices are negatives unless explicit negative indices are
provided. Probes fit the training set and rank heads by validation accuracy.
Following [ITI](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html),
the selected head directions use the truthful-minus-false means from both
development splits, normalized and multiplied by their projection standard
deviation. Unselected head slices remain zero. The returned vector preserves
this scale; apply it with `algorithm="attention_add"`, `normalize=False`, and
`scale` set to the desired intervention strength. The generic extraction
interface also accepts `method="iti"` with the same keyword arguments.

The [ITI notebook](https://github.com/ZJU-REAL/EasySteer/blob/main/replications/iti/iti.ipynb)
demonstrates Llama-2-7B-Chat with a supplied direction: 48 heads, strength 15,
one fixed TruthfulQA evaluation set, an accuracy comparison, and generated
examples. It uses the authors' QA prompt and loads the model once. The supplied
direction uses the official code's separate, unlabeled GEN calibration bank;
the extractor above uses its training and validation captures for that scale.

## Learning-based: `easysteer.reft`

Reimplements pyreft: trains a parameterized intervention (e.g. `BiasIntervention`,
LoReFT) on a frozen HuggingFace model with a standard `transformers` trainer, then saves
the learned representation for inference.

Use `easysteer.reft.train.train_reft` for the shared training pipeline. Bias
checkpoints are loaded through `easysteer.vectors.from_pyreft` and applied with
`algorithm="direct"`; LoReFT checkpoints use the same adapter with
`algorithm="loreft"`. These checkpoints are passed through `VectorSpec(data=...)`,
rather than as a third-party checkpoint path in `source`.

The complete training walkthrough (data module, trainer, saving) is in
[ReFT training](reft-training.md); see the
[LoReFT replication](../replications/index.md) for a complete train-then-steer
notebook.
