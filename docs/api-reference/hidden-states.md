# `easysteer.capture`

Hidden-state, attention-head, and MoE router-logit capture from a running vLLM
engine. The [capture guide](../user-guide/hidden-state-capture.md) explains
selection, graph execution, and prefix-cache behavior.

## Capture

::: easysteer.capture.capture

::: easysteer.capture.capture_batches

::: easysteer.capture.release_capture_cache

::: easysteer.capture.CaptureResult
    options:
      merge_init_into_class: true
      members: [layers, outputs, layouts, component, model, selection, per_prompt_selections, sample_indices, layer_ids, labelled, rows, meta, sample, sample_rows, iter_sample_rows, token, save, load, sample_positions, sample_token_ids, to_nested, __len__]
      show_if_no_docstring: true

`capture()` returns a `CaptureResult`. Its methods select rows by true layer ID
or by input sample; `len(result)` is the number of generated request outputs.
An empty selection still preserves the outputs and empty per-sample views.
`sample_rows(i, layer)`, `token(i, layer)` and the other sample methods use local
indices within this result. `sample_indices[i]` identifies that sample in the
original input iterable when using `capture_batches()`. `selection` records the
global clause; `per_prompt_selections` records optional overrides, with `None`
entries falling back to the global clause. Both survive disk save/load.
Row-label types, component constants, serialization, and worker stream/session
classes are documented in the [fork capture API](capture.md).
