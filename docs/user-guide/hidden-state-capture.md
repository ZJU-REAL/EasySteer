# Hidden-state capture

`easysteer.hidden_states` extracts intermediate activations from a running vLLM
engine for analysis, training data and steering-vector extraction. Capture works
without enabling steering.

## Engine requirements

Capture requires the V2 GPU model runner, which vLLM 0.29 selects by default.
Models or features that fall back to V1, and standalone multimodal encoder
runners, cannot use capture. Steering and capture currently require token-ID
prompts; prompt embeddings are not supported.

```python
from vllm import LLM
import easysteer.hidden_states as hs

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", tensor_parallel_size=1)
# Capture chooses its execution path automatically. Admission skips
# prefix-cache reads when necessary to compute selected prompt rows.
```

The client helper requires a single worker. Tensor-parallel capture returns
per-rank shards and is rejected rather than merging them implicitly. Capture
also depends on the engine being able to discover the model's decoder layers
or MoE gates; test new model architectures before relying on their activations.

## Capture a batch

One call captures a batch of prompts and returns a labelled
[`CaptureResult`](../api-reference/hidden-states.md). For prompt-only analysis,
`max_tokens=1` runs the prompt forward pass without subsequent decode steps:

```python
tokenizer = llm.get_tokenizer()
prompts = []
for content in ["What is steering?", "Explain PCA."]:
    messages = [{"role": "user", "content": content}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True,
    )
    prompt = {"prompt_token_ids": prompt_ids}
    prompts.append(prompt)
result = hs.capture(
    llm,
    prompts=prompts,
    max_tokens=1,          # 1 = prompt-only forward pass
    layers=[10, 11, 12],   # true layer ids; None = all hooked layers
    dtype="float16",       # engine-side storage dtype (optional)
)
```

Key arguments (full signature in the [API reference](../api-reference/hidden-states.md)):

| Argument | Meaning |
|---|---|
| `max_tokens` | Tokens to generate; `1` captures only the prompt forward pass. |
| `layers` | Layer-id subset (`None` = all). Layers are keyed by **true layer id** everywhere, never positional index. |
| `dtype` | Engine-side storage dtype: `"float16"`, `"bfloat16"`, `"float32"`, `"float64"`, `"int32"` or `"int64"`. `None` preserves the model output dtype. |
| `select` | Global `SelectSpec` (or wire dict) choosing which rows to keep. |
| `per_prompt_selects` | One `SelectSpec` or wire dict per prompt, overriding the global selection (`None` entries keep the global one). The list length must match `prompts`. |
| `stream` | `"hidden_states"` (default), `"router_logits"` (MoE), or `"attention_heads"` (before the attention output projection). |
| `steering` | `SteeringSpec` or per-prompt list, as in `LLM.generate()`. Captured values include steering; use `False` to disable it even when the engine has a default. |
| `budget_bytes` | Optional limit on raw CPU values and row labels, including pending transfers, across the stream's layers. Exceeding it fails capture instead of returning a partial result. |
| `**generate_kwargs` | Forwarded into `SamplingParams` (e.g. `temperature`). |

### Select rows

Pass a `SelectSpec` to collect only the rows needed for analysis. It shares
[steering's selection rules](steering.md#select-prompt-and-generation-rows):
select each phase independently, combine includes by union, and subtract
exclusions. For one representation per prompt, select its last token:

```python
from vllm.capture import SelectSpec

# Keep only the last prompt token of each sample
result = hs.capture(llm, prompts,
                    select=SelectSpec(prompt_positions=[-1]))
```

The helper returns selected rows without engine-side reduction. It does not
accept `positions` or `reduce` arguments: express positions in `SelectSpec`, then
reduce the returned tensors if needed. With `max_tokens=1`, generation-only
selectors have no decode-forward rows to capture. Use a longer generation when
selecting decode rows. An empty selection returns no layers and empty sample
views.

## Read captured rows

Rows are grouped by their owning request via engine labels and ordered by sequence
position — the only correct grouping under continuous batching.

```python
result.layer_ids          # sorted true layer ids
result.rows(12)           # Tensor(total_rows, dim) for layer 12, all samples
result.sample(0)          # {layer_id: Tensor(rows, dim)} for sample 0
result.sample_rows(0, 12) # sample 0's rows at layer 12 only
result.token(0, 12, -1)   # last captured row of sample 0 at layer 12
result.sample_positions(0)  # absolute sequence positions of sample 0's rows
result.sample_token_ids(0)  # input token ids of sample 0's rows
result.outputs            # the vLLM RequestOutput list, prompt order
result.layouts            # per-layer component widths and attention head layout
result.to_nested()        # [sample][layer_pos] tensors, when explicitly needed
```

`result.meta(layer)` exposes the raw row labels (`req_ids` / `positions` / `token_ids`).
Labels are mandatory. `token()` indexes a sample's captured rows, so `-1`
means the last selected row, which may differ from the last original prompt
position. `sample_rows()` returns a view when those rows are contiguous;
noncontiguous rows are gathered in sequence order.

## Process a dataset in batches

Use `capture_batches()` to consume results a batch at a time. It preserves
prompt order and slices per-prompt steering and selection lists along the same
boundaries. Each result uses local sample indices starting at zero. Worker
storage is cleared before the result is yielded, and compatible capture graphs
remain cached for the next batch.

For diffmean, keep running statistics rather than all captured samples:

```python
from easysteer.steer import DiffMeanAccumulator, DiffMeanExtractor

accumulator = DiffMeanAccumulator()
for positive, group in [(True, positive_prompts), (False, negative_prompts)]:
    for batch in hs.capture_batches(
        llm, group, batch_size=32, layers=[10, 11, 12],
        select=SelectSpec(prompt_positions=[-1]),
        budget_bytes=256 * 1024 * 1024,
    ):
        for layer in batch.layer_ids:
            accumulator.update(layer, batch.rows(layer), positive=positive)
        del batch

vector = DiffMeanExtractor.from_moments(accumulator.pos, accumulator.neg)
vector.export_gguf("direction.gguf")
```

`positive_prompts` and `negative_prompts` use the same prompt input format as
`capture()`. Selecting one row per prompt gives each sample equal weight. For
full activation datasets, write each yielded batch to a separate file instead
of retaining the iterator as a list. One large batch can still exceed the byte
budget; reduce its size, select fewer rows/layers, or increase the limit.

`capture_batches()` defaults to 32 prompts and a 256 MiB raw-storage budget.
`capture()` retains its unrestricted default, with `budget_bytes` available
explicitly. The budget counts values and labels in worker CPU storage and
pending transfers. It excludes model memory, CUDA graph buffers, serialization
copies, and tensors retained by the caller; it is not a process memory limit.
The low-level `capture_status` RPC reports `storage_bytes` and `budget_bytes`.

Per-layer extraction reads only the requested token rows and processes one
layer at a time. Online means require space proportional to feature width.
`MomentsAccumulator(track_second_moment=True)` also retains a square covariance
matrix per layer; use it only when that statistic is needed. It is not a general
memory-saving replacement for sample storage at large hidden dimensions.

## MoE router logits

Pass `stream="router_logits"` to `hs.capture()` on an MoE model to capture per-token
router logits instead of hidden states (used e.g. by the
[SteerMoE replication](../replications/index.md)).

## Attention head outputs

`stream="attention_heads"` captures attention aggregation outputs before the
output projection, with the same selection and per-sample views:

```python
from vllm.capture import SelectSpec

heads = hs.capture(
    llm, prompts, layers=[10], stream="attention_heads",
    select=SelectSpec(prompt_positions=[-1]), steering=False,
)
layout = heads.layouts[10]
head_outputs = heads.sample(0)[10].reshape(
    -1, layout["num_heads"], layout["head_size"],
)
```

The stored tensors remain two-dimensional `(rows, width)`. The layout records
`width = num_heads * head_size`, where `num_heads` is the query head count and
`head_size` is the value-output dimension per head. Neither KV head counts nor
the model's residual hidden size should be used to infer this layout.

This stream supports standard decoder MHA/GQA with `tensor_parallel_size=1`;
MLA, encoder attention, and cross-attention are not exposed as head outputs.
It shares capture's graph and prefix-cache policy. These activations feed the
[ITI extractor](extracting-vectors.md#iti-attention-head-directions).

## Multimodal inputs

Multimodal prompts use the same list of input dictionaries as `LLM.generate`,
including `prompt` and `multi_modal_data`; capture preserves their cache salts.

## Graph execution

Eligible FULL-graph batches use a separate CUDA graph that records activations.
This path requires one worker, no LoRA or speculative decoding, and steering
either disabled or running `in_graph`. Other batches that need captured rows
run eagerly; steps with no selected rows keep ordinary model execution.
The first eligible batch records the capture graph. One component/layer
combination stays cached across `capture()` calls; changing the selected token
positions, storage dtype or reduction does not require recording it again.
Changing components or layers replaces that cached graph. Ordinary model graphs
do not contain capture operations.

The graph's fixed output buffers contain whole execution batches before row
selection. Selecting fewer rows reduces retained CPU data and transfer volume,
but does not shrink those fixed GPU buffers. Stopping capture clears the stream
while retaining the cached graph for reuse; status reports its buffer and total
allocation sizes separately.

## Prefix caching

KV blocks do not contain the intermediate activations skipped by a prefix
hit. Request admission therefore sets `skip_reading_prefix_cache=True` when
the effective selection requires those prompt rows. Generation-only selection,
the last prompt token, and other selections proven unaffected by prefix hits
can reuse cached blocks. Per-prompt overrides participate in this decision.

This policy applies both to `hs.capture()` and to requests submitted through the
same engine after a successful `start_capture` RPC. Cache writes remain enabled
under the ordinary cache key, so subsequent matching requests can reuse them;
capture does not add a random salt. Existing steering fingerprints still isolate
different interventions. If capture starts after a request has already reused
selected prompt rows, fetching that incomplete capture raises an error. The check
uses the stream's selection and any per-request override; selections confined to
uncached rows can still be captured. Existing requests are checked when they
first run in a batch after capture starts, so a completed request awaiting worker
cleanup does not affect the new stream.

If steering fails for a request, fetching its captured rows also reports that
failure. Other requests remain available through a fetch restricted to their
request IDs; clearing the stream removes its retained rows and errors.

Pass `CaptureResult` directly to vector extractors when capturing a layer
subset. Converting it to a plain nested list loses the mapping from list
positions to true layer IDs. The older `get_all_hidden_states_generate` and
`get_moe_router_logits_generate` wrappers have been removed; use `capture()`
with the appropriate `stream`, then `result.outputs`, `result.layers`, or
`result.to_nested()` when that representation is required.
