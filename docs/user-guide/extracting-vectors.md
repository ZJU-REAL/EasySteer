# Extracting steering vectors

Two complementary routes turn captured hidden states into interventions.

## Analysis-based: `easysteer.steer`

Computes a semantic direction from contrastive hidden states — no training loop.
Available extractors: DiffMean, PCA, LAT, linear probe, and SAE feature vectors.

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
