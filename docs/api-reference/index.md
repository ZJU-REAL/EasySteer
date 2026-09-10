# API Reference

This reference documents EasySteer's Python interfaces and its additions to
vLLM 0.29.0. Use the linked guides for complete workflows; class pages provide
signatures, field types, defaults, and return values from the source.

## Engine configuration

- [Engine configuration](engine-configuration.md): steering startup arguments,
  CLI equivalents, and concurrent configuration capacity.
- [Algorithms and payloads](algorithms.md): supported components, graph
  conditions, transformations, and file formats.

General model-loading and sampling options remain upstream vLLM interfaces.
See the upstream [LLM API](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/)
and [engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/),
using documentation that matches your installed vLLM version.

## Steering

| Interface | Use it to | Guide |
|---|---|---|
| [SteeringSpec, VectorSpec, ApplySpec](steering-specs.md) | Describe an intervention and attach it to a request or engine default. | [Steering requests](../user-guide/steering.md) |
| [Payload classes](payloads.md) | Construct validated direction, map, low-rank, concept-pair, or router weights. | [Load your weights](../user-guide/steering.md#steering-with-your-own-tensors) |
| [LLM default and preload methods](steering-specs.md#engine-default-and-preload-methods) | Update the default for new requests or preload file-backed weights. | [Serving and management](../user-guide/openai-server.md#management-endpoints) |

## Capture

| Interface | Use it to | Guide |
|---|---|---|
| [capture() and CaptureResult](hidden-states.md) | Collect labelled activations and select rows by input sample. | [Hidden-state capture](../user-guide/hidden-state-capture.md) |
| [SelectSpec](steering-specs.md#model_hooks.selection.spec.SelectSpec) | Select prompt and generation rows with the shared selection language. | [Select rows](../user-guide/steering.md#select-prompt-and-generation-rows) |
| [vllm.capture](capture.md) | Integrate row metadata, serialization, and worker stream/session types. | [Capture execution](../user-guide/hidden-state-capture.md#graph-execution) |

## Extraction and extension

- [Vector extraction and adapters](steer.md): statistical extractors, ITI, SAE,
  `StatisticalControlVector`, and `easysteer.vectors` checkpoint adapters.
  Start with [extracting vectors](../user-guide/extracting-vectors.md).
- [Algorithm extension API](algorithm-extension.md): transformation and graph
  contracts, registration, and authoring capabilities. Follow the
  [contribution guide](../developer-guide/contributing.md#adding-a-steering-algorithm)
  when implementing an algorithm.
