# Extracting steering vectors

Two complementary routes turn captured hidden states into interventions.

## Analysis-based: `easysteer.steer`

Computes a semantic direction from contrastive hidden states — no training loop.
Available extractors: DiffMean, PCA, LAT, linear probe, and SAE feature vectors.

```python
from easysteer.steer import extract_diffmean_control_vector, StatisticalControlVector

control_vector = extract_diffmean_control_vector(
    all_hidden_states=all_hidden_states,  # nested [samples][layer][token]
    positive_indices=[0, 1, 2, 3],
    negative_indices=[4, 5, 6, 7],
    model_type="qwen2.5",
    token_pos=-1,       # which token's activation to use
    normalize=True,
)

control_vector.export_gguf("vectors/diffmean.gguf")
# ... later
control_vector = StatisticalControlVector.import_gguf("vectors/diffmean.gguf")
```

The exported GGUF file is what `VectorSpec(source=...)` consumes at inference time.

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

```python
import easysteer.reft as reft

reft_config = reft.ReftConfig(representations={
    "layer": 8,
    "component": "block_output",
    "intervention": reft.BiasIntervention(embed_dim=model.config.hidden_size),
})
reft_model = reft.get_reft_model(model, reft_config)
# ... build a data module, run reft.ReftTrainer, then reft_model.save(...)
```

See the [LoReFT replication](../replications/index.md) for a complete train-then-steer
notebook.

<!-- TODO: dedicated pages for each extractor (assumptions, when to prefer which),
GGUF file format description, and the reft training API surface. -->
