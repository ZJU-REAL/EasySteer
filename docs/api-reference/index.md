# API Reference

Generated with [mkdocstrings](https://mkdocstrings.github.io/) from the source
docstrings. Building these pages requires `easysteer` (and for the steering-spec page,
the `vllm-steer` fork) to be importable in the docs environment.

- [`easysteer.steer`](steer.md) — vector extraction (DiffMean, PCA, LAT, linear probe,
  SAE) and the `StatisticalControlVector` container.
- [`easysteer.hidden_states`](hidden-states.md) — `capture()` / `CaptureResult` and the
  legacy extraction helpers.
- [Steering specs](steering-specs.md) — `SteeringSpec` / `VectorSpec` / `ApplySpec` /
  `SelectSpec` from `vllm.steer_vectors.api`.

<!-- TODO: consider vLLM-style api-autonav for full-module auto-navigation once the
docstring coverage of easysteer.steer / easysteer.reft is raised. easysteer.reft is
not yet included here. -->
