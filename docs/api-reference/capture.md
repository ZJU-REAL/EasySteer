# Capture API (`vllm.capture`)

Applications normally call
[`easysteer.hidden_states.capture()`](hidden-states.md#easysteer.hidden_states.capture),
which starts capture, generates requests, fetches labelled tensors, and stops the
stream. The fork exports the selection, serialization, and worker-session types
documented below for integrations.

```python
from vllm.capture import (
    ATTENTION_HEADS,
    HIDDEN_STATES,
    ROUTER_LOGITS,
    CaptureMeta,
    SelectSpec,
    deserialize_captured,
    match_capture_request_id,
)
```

## Components and selection

| Constant | Value | Captured representation |
|---|---|---|
| `HIDDEN_STATES` | `"hidden_states"` | Decoder-layer output; models returning a separate residual are combined for capture. |
| `ATTENTION_HEADS` | `"attention_heads"` | Concatenated query-head outputs before the attention output projection. |
| `ROUTER_LOGITS` | `"router_logits"` | Accessible MoE gate scores before routing. |

[`SelectSpec`](steering-specs.md#model_hooks.selection.spec.SelectSpec)
defines which prompt and generation rows to collect. It is the same class
exported from `vllm.steer_vectors`; its field reference is rendered once on the
specs page. See the [capture guide](../user-guide/hidden-state-capture.md) for
model support and graph/cache behavior.

## Row labels and deserialization

The decoded layer tensors and their `CaptureMeta` labels have matching row
counts. Request labels identify the owning request under continuous batching;
sequence positions order rows within a sample.

::: model_hooks.capture.serialization.CaptureMeta
    options:
      heading: vllm.capture.CaptureMeta
      merge_init_into_class: true
      members: [req_ids, positions, token_ids, __len__]
      show_if_no_docstring: true

::: model_hooks.capture.serialization.deserialize_captured
    options:
      heading: vllm.capture.deserialize_captured

`serialized_data` is the per-layer dictionary returned by a worker's
`fetch_captured` RPC. The function returns `(tensors, meta)`, both keyed by true
layer ID. Optional attention layout metadata stays in the original wire
dictionary; the application helper attaches it to `CaptureResult.layouts`.

::: model_hooks.capture.serialization.match_capture_request_id
    options:
      heading: vllm.capture.match_capture_request_id

Use this helper to compare an engine row label with a client request ID. It
accepts an exact match or the engine's recognized uniqueness suffix and returns
a boolean; arbitrary prefix matching can mix different requests.

## Worker stream configuration

These classes are exported for engine integrations. Ordinary callers should
use `capture()` instead of creating or attaching a second `CaptureSession`.

`StreamConfig` is constructed when a session enables a stream:

| Parameter | Meaning |
|---|---|
| `layers` | True layer IDs; `None` selects all available layers. |
| `dtype` | Optional storage dtype, independent of model compute dtype. |
| `reduce` | `"all"` retains rows; `"last"` or `"mean"` reduces within a request. |
| `select` | A `SelectSpec.to_wire()` dictionary; requires `reduce="all"`. |
| `budget_rows` | Optional nonnegative row limit per layer. |

The constructor validates the selection and resolves dtype/layer storage. These
low-level reduction and budget arguments are not arguments to the high-level
`hs.capture()` helper, which returns selected rows without a reduction.

::: model_hooks.capture.store.StreamConfig
    options:
      heading: vllm.capture.StreamConfig
      merge_init_into_class: true
      members: [layers, dtype, reduce, select, budget_rows, selects_rows]
      show_if_no_docstring: true

## Capture session

The model runner owns a `CaptureSession`, attaches the discovered components at
model load, and calls its lifecycle methods for requests and batches. For stream
management, the relevant operations are:

| Method | Effect / result |
|---|---|
| `enable_stream(stream, **config_kwargs)` | Validate a `StreamConfig` and start a fresh store; hooks must already be attached. |
| `disable_stream(stream)` | Disable collection for that stream. |
| `fetch_stream(stream, clear=True, layers=None, req_ids=None)` | Return serialized rows, labels, and available layouts; optionally clear the selected data. |
| `clear_stream(stream)` | Clear retained rows and errors. |
| `stream_status(stream)` | Return enablement, hooked layers, layouts, storage counters, and capture-graph execution counters. |

The supported RPCs (`start_capture`, `stop_capture`, `fetch_captured`,
`clear_captured`, `capture_status`) delegate to these operations in the worker.
The high-level helper also coordinates prefix-cache read policy before admission.

::: model_hooks.capture.session.CaptureSession
    options:
      heading: vllm.capture.CaptureSession
      merge_init_into_class: true
      members: [attach, detach, enable_stream, disable_stream, fetch_stream, clear_stream, stream_status]
      show_if_no_docstring: true

## Stream store

`StreamStore(config)` holds one stream's captured rows and metadata. A session
owns it; integrations can inspect the storage/serialization contract here.
`serialize()` returns the per-layer wire dictionaries consumed by
`deserialize_captured()`. Request and layer filters operate on retained data.

::: model_hooks.capture.store.StreamStore
    options:
      heading: vllm.capture.StreamStore
      merge_init_into_class: true
      members: [tokens_stored, append, flush, serialize, clear, drop_layers]
      show_if_no_docstring: true
