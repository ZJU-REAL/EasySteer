# API Reference

The `easysteer` pages are generated with
[mkdocstrings](https://mkdocstrings.github.io/) from the source docstrings via static
analysis (`allow_inspection: false`): the build parses the source tree and never
imports it, so it runs in a plain environment with only `requirements-docs.txt` — no
torch or vLLM install required.

- [`easysteer.steer`](steer.md) — vector extraction (DiffMean, PCA, LAT, linear probe,
  SAE) and the `StatisticalControlVector` container.
- [`easysteer.hidden_states`](hidden-states.md) — `capture()` / `CaptureResult` and the
  legacy extraction helpers.
- [Steering specs](steering-specs.md) — hand-written summary of `SteeringSpec` /
  `VectorSpec` / `ApplySpec` / `SelectSpec`; the
  [Steering guide](../user-guide/steering.md) is the canonical reference (rendering the
  `vllm-steer` fork with mkdocstrings would drag the CUDA/torch stack into the docs
  build).

<!-- TODO: consider vLLM-style api-autonav for full-module auto-navigation once the
docstring coverage of easysteer.steer / easysteer.reft is raised. easysteer.reft is
not yet included here. -->
