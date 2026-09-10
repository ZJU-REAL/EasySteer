# API Reference

Use the task guides for complete workflows and these pages to look up arguments,
formats, and signatures.

- [Engine configuration](engine-configuration.md) — startup arguments, CLI flags,
  graph selection, and concurrent configuration capacity.
- [Algorithms and payloads](algorithms.md) — component and graph support,
  transformation semantics, native sources, checkpoint adapters, and JSON.
- [`easysteer.steer`](steer.md) — vector extraction (DiffMean, PCA, LAT, linear probe,
  ITI, SAE), the `StatisticalControlVector` container, and payload adapters.
- [`easysteer.hidden_states`](hidden-states.md) — `capture()` / `CaptureResult` and the
  legacy extraction helpers.
- [Steering specs](steering-specs.md) — generated class and field reference for
  `SteeringSpec`, `VectorSpec`, `ApplySpec`, and `SelectSpec`, plus default and
  preload methods added to `LLM`.
- [Payload classes](payloads.md) — constructors, tensor shapes, validation,
  attributes, and wire conversion for all six payload types.
- [Capture API](capture.md) — `vllm.capture` component constants, row metadata,
  serialization, and worker stream/session classes.
- [Algorithm extension API](algorithm-extension.md) — transformation and graph
  contracts, registration, and authoring capabilities.

Class and method references use static analysis of the checked-out EasySteer and
vLLM-fork sources. Public lazy exports are rendered from their defining modules
with the public import names in their headings. Building the site does not
import either package or require a GPU runtime.
