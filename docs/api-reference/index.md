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
- [Steering specs](steering-specs.md) — hand-written summary of `SteeringSpec` /
  `VectorSpec` / `ApplySpec` / `SelectSpec`; the
  [Steering guide](../user-guide/steering.md) is the canonical reference. These
  classes belong to the vLLM fork and are documented separately from the generated
  `easysteer` package reference.

The `easysteer` API pages are generated from source docstrings with static
analysis. Fork specs and engine options are documented separately so the site
can build without importing a GPU runtime.
