# `easysteer.hidden_states`

Hidden-state, attention-head, and MoE router-logit capture from a running vLLM
engine. The [capture guide](../user-guide/hidden-state-capture.md) explains
selection, graph execution, and prefix-cache behavior.

## Capture (recommended)

::: easysteer.hidden_states.capture

::: easysteer.hidden_states.CaptureResult
    options:
      merge_init_into_class: true
      members: [layers, outputs, layouts, layer_ids, labelled, rows, meta, sample, sample_positions, sample_token_ids, to_nested, __len__]
      show_if_no_docstring: true

`capture()` returns a `CaptureResult`. Its methods select rows by true layer ID
or by input sample; `len(result)` is the number of generated request outputs.
An empty selection still preserves the outputs and empty per-sample views.
Row-label types, component constants, serialization, and worker stream/session
classes are documented in the [fork capture API](capture.md).

## Compatibility wrappers

Nested-list wrappers over `capture()`; prefer `capture()` for new code.

::: easysteer.hidden_states.get_all_hidden_states_generate

::: easysteer.hidden_states.get_moe_router_logits_generate
