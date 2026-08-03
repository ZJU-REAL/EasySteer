# `easysteer.steer`

Analysis-based extraction of steering vectors from captured hidden states.

## Unified extraction interface

::: easysteer.steer.extract_statistical_control_vector

::: easysteer.steer.extract_diffmean_control_vector

::: easysteer.steer.extract_pca_control_vector

::: easysteer.steer.extract_lat_control_vector

::: easysteer.steer.extract_linear_probe_control_vector

## Containers and utilities

::: easysteer.steer.StatisticalControlVector

::: easysteer.steer.extract_token_hiddens

## SAE helpers

::: easysteer.steer.search_sae_features

::: easysteer.steer.get_sae_feature_explanation

::: easysteer.steer.extract_sae_decoder_vector

<!-- TODO: add the extractor classes (DiffMeanExtractor, PCAExtractor, LATExtractor,
LinearProbeExtractor, SAEFeatureExplorer) and accumulators once their docstrings are
reviewed. -->

## Payload adapters (`easysteer.vectors`)

Client-side adapters from third-party checkpoint formats to the canonical
steering payloads passed via `VectorSpec(data=...)`.

::: easysteer.vectors.from_control_vector

::: easysteer.vectors.from_gguf

::: easysteer.vectors.from_pt_direction

::: easysteer.vectors.from_pyreft

::: easysteer.vectors.from_lm_steer

::: easysteer.vectors.from_linear_transport
