# Capture API (`vllm.capture`)

Applications normally call
[`easysteer.capture.capture()`](hidden-states.md#easysteer.capture.capture),
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
    assemble_captured,
    deserialize_captured,
    match_capture_request_id,
    validate_capture_topology,
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
dictionary. With TP, this function decodes one worker's local values; use
`assemble_captured()` to obtain complete tensors and global layouts.

::: model_hooks.capture.serialization.assemble_captured
    options:
      heading: vllm.capture.assemble_captured

Pass the complete list of `fetch_captured` worker results and `tp_size`. The
function returns `(tensors, meta, layouts)`, keyed by true layer ID. Replicated
hidden states and router logits are exported only by TP rank 0; other workers
return no layer data for those streams. Attention workers export feature shards
with local `layout` and explicit `shard` metadata: `kind`, `tp_rank`, `tp_size`,
`feature_start`, and `global_width`.

Assembly validates shard coverage and matching request IDs, positions, and
token IDs, then joins attention values in global query-head order. Worker reply
order does not determine head order. The application helper uses this function
and attaches the resulting global layouts to `CaptureResult.layouts`.

::: model_hooks.capture.serialization.validate_capture_topology
    options:
      heading: vllm.capture.validate_capture_topology

Call this with all worker `capture_status` results before starting a raw session.
It returns the TP size after checking that replies form one complete TP group:
`PP=DP=1`, prefill/decode context parallelism disabled, and no sequence or expert
parallelism. Stream lifecycle RPCs still run on every worker, including workers
that do not export replicated values. Check every worker's result and status;
selecting only the first reply can miss a capture error.

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
| `budget_bytes` | Optional raw CPU values/labels and pending-transfer limit. `start_capture` treats it as a total across layers and workers. Overflow is reported at fetch; it does not fail the model forward. |

The constructor validates the selection and resolves dtype/layer storage. With
TP, the session divides an attention stream's byte budget equally among workers,
including each worker's labels, rounding down to whole bytes. Replicated streams
retain the full budget on their owner. A worker's `StreamConfig` and status show
its assigned limit, so this is a conservative total bound with no redistribution
of unused capacity. The high-level `hs.capture()` accepts `budget_bytes` but not
`budget_rows` or `reduce`; it returns selected rows without a reduction.

::: model_hooks.capture.store.StreamConfig
    options:
      heading: vllm.capture.StreamConfig
      merge_init_into_class: true
      members: [layers, dtype, reduce, select, budget_rows, budget_bytes, selects_rows]
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
| `stream_status(stream)` | Return enablement, hooked layers, local layouts, storage counters, and capture-graph execution counters. |

The supported RPCs (`start_capture`, `stop_capture`, `fetch_captured`,
`clear_captured`, `capture_status`) delegate to these operations in the worker.
The worker adds topology metadata to `capture_status`. The high-level helper
also coordinates prefix-cache read policy before admission. Eligible FULL
batches use a separate capture graph at both `TP=1` and `TP>1`, with steering
disabled or `in_graph` and without LoRA or speculative decoding. Other selected
steps use eager forwards. Graph replay counters advance on every TP rank,
including non-owners with no local capture buffers.

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
