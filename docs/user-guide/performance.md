# Graphs, caching, and performance

Start with the engine defaults and declare the algorithms your requests need.
EasySteer selects how steering integrates with compiled execution. Prefix
caching and chunked prefill remain available without extra steering flags.

## Graph modes

EasySteer's `steer_graph_mode` and vLLM's CUDA graph mode describe different
parts of execution:

| Setting | Steering execution | When to use it |
|---|---|---|
| `steer_graph_mode="auto"` | Resolves to one of the next two modes at startup. | Default for new applications. The startup log includes the reason. |
| `steer_graph_mode="in_graph"` | Steering kernels read persistent payload/selection buffers inside compiled execution. | Single-vector requests whose algorithms and payloads meet the graph conditions. |
| `steer_graph_mode="split"` | The model runs in compiled segments, with steering between them. | Multi-vector requests, `linear`, or payloads outside the in-graph conditions. |
| `enforce_eager=True` | Model and steering execute without CUDA graph capture. | Debugging or short jobs where compilation startup dominates total work. |

`split` retains graph acceleration for model segments. `in_graph` keeps steering
eligible for full CUDA graphs, but the actual vLLM graph mode also depends on the
model, attention backend, and other engine features. Check the engine's CUDA
graph logs as well as the steering-mode log. See the upstream
[CUDA graph design](https://docs.vllm.ai/en/latest/design/cuda_graphs/) for the
model execution modes.

An explicit `in_graph` request is checked rather than silently downgraded. It
cannot be combined with `enforce_eager=True`. Changing vectors, scales, or token
selection updates persistent buffers; it does not require restarting the model.

### How auto chooses

With compilation enabled, auto first checks a concrete startup default if one
is supplied. A single-vector default that fits the graph buffers selects
`in_graph`; a multi-vector default or unsupported payload selects `split`.
Later requests must fit the mode selected at startup.

Without a startup default, auto uses the declaration:

1. A declaration allowing multiple vectors selects `split`.
2. Single-vector declarations of `direct`, `attention_add`, `erase`, `replace`,
   and `concept_replace` select `in_graph`.
3. LoReFT and LM-Steer require a rank bound; MoE routing requires a supported
   routing mode. Names alone do not establish these payload conditions, so
   these declarations select `split`.
4. `linear` has no in-graph kernel and selects `split`.

An engine that must accept a bounded conditional workload can explicitly select
`in_graph`; request admission then enforces its conditions. The
[algorithm table](../api-reference/algorithms.md) lists these conditions and
normalization support. Normalization does not require eager mode for algorithms
that support it.

## Prefix caching

For generation, KV-cache keys include the effective steering configuration.
Identical prompts and steering configurations can reuse cached blocks. A
baseline request with `steering=False` and a steered request use separate keys.
Updating a default preserves cache entries for previous configurations.

Capture needs activations that KV blocks do not store. When selected prompt rows
would be skipped by a prefix hit, admission bypasses cache **reads for that
request**. Cache writes remain enabled. Last-prompt-token and generation-only
selections can reuse prefix blocks when their selected rows are still computed.
Capture uses no random cache salt. See the complete
[capture cache policy](hidden-state-capture.md#prefix-caching).

Chunked prefill uses the same selection semantics. Negative prompt positions
resolve against the full prompt length, so `prompt_positions=[-1]` consistently
selects its last token across chunks. Disabling chunked prefill or prefix caching
globally is unnecessary for ordinary steering and capture.

## Capture execution

Capture has a separate graph containing the selected component/layer taps.
Eligible FULL-graph batches reuse it across capture calls. Changing row
selectors or storage dtype does not require a new graph; changing the component
or layer set replaces the cached capture graph.

This graph path requires one GPU worker, no LoRA or speculative decoding, and
steering disabled or using `in_graph`. Other batches with selected rows use
eager capture. A step with no selected rows follows normal model execution.
The [capture guide](hidden-state-capture.md#graph-execution) describes these
requirements; an eager capture step does not mean ordinary generation has lost
its graphs.

Capture only the layers and rows needed for analysis. For one representation per
prompt, use `SelectSpec(prompt_positions=[-1])` and `max_tokens=1`. This reduces
activation storage and transfer as well as avoiding unnecessary decode steps.

## Choosing settings for a workload

- **A persistent demo or server:** keep the model loaded and use graph defaults.
  Declare only the algorithms the service accepts. Preload frequently used
  payloads when request latency matters.
- **Repeated evaluations:** load the model once, batch prompts, and reuse it
  for baseline and steered requests. Keep the paper's prompt and selection
  protocol when reproducing results.
- **A small, one-off capture job:** eager mode can finish sooner if graph
  compilation costs more than the inference it would save. Keep the choice
  explicit at that job rather than making it the shared model-loader default.

## Measuring performance

Separate initialization from request execution. The first engine start can
include model loading, kernel compilation, and CUDA graph capture. Warm the
execution path before comparing steady-state latency or throughput.

For comparable requests, record prompt-token counts, generated-token counts,
sampling settings, batch size, graph mode, and whether prefix-cache hits are
included. `max_tokens` is an upper bound; generated answers may end earlier.
Temperature zero is useful for reducing output-length variation in a timing
comparison, but the actual token counts are the quantities to report.

The repository's
[`bench_eager_vs_cudagraphs.py`](https://github.com/ZJU-REAL/EasySteer/blob/main/tests/bench_eager_vs_cudagraphs.py)
compares execution modes. Use a workload representative of your application;
the paper's reported speedups describe its own models, hardware, and baselines.
