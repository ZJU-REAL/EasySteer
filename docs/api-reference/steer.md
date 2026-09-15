# `easysteer.extraction`

Analysis-based extraction of steering vectors from captured hidden states.

## Unified extraction interface

::: easysteer.extraction.extract

::: easysteer.extraction.extract_statistical_control_vector

::: easysteer.extraction.extract_diffmean_control_vector

::: easysteer.extraction.extract_pca_control_vector

::: easysteer.extraction.extract_lat_control_vector

::: easysteer.extraction.extract_linear_probe_control_vector

## ITI attention extraction

::: easysteer.extraction.ITIExtractor

## Containers and utilities

::: easysteer.extraction.StatisticalControlVector

::: easysteer.extraction.extract_token_hiddens

## SAE helpers

::: easysteer.extraction.search_sae_features

::: easysteer.extraction.get_sae_feature_explanation

::: easysteer.extraction.extract_sae_decoder_vector

## Payload adapters (`easysteer.vectors`)

Client-side adapters from third-party checkpoint formats to the canonical
steering payloads passed via `VectorSpec(data=...)`.

::: easysteer.vectors.load

::: easysteer.vectors.to_json_payload

::: easysteer.vectors.from_control_vector

::: easysteer.vectors.from_gguf

::: easysteer.vectors.from_pt_direction

::: easysteer.vectors.from_training

::: easysteer.vectors.from_pyreft

::: easysteer.vectors.from_lm_steer

::: easysteer.vectors.from_linear_transport
