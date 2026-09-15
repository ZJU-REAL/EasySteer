# Hidden-state capture

`easysteer.capture` extracts intermediate activations from a running vLLM
engine for analysis, training data and steering-vector extraction. Capture works
without enabling steering.

## Engine requirements

Capture requires the V2 GPU model runner, which vLLM 0.29 selects by default.
Models or features that fall back to V1, and standalone multimodal encoder
runners, cannot use capture. Steering and capture currently require token-ID
prompts; prompt embeddings are not supported.

```python
from vllm import LLM
import easysteer.capture as hs

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", tensor_parallel_size=1)
# Capture chooses its execution path automatically. Admission skips
# prefix-cache reads when necessary to compute selected prompt rows.
```

The client helper supports ordinary tensor parallelism, including
`tensor_parallel_size=2` on two GPUs. Use one pipeline stage and one data-parallel
replica (`PP=DP=1`), with context, sequence, and expert parallelism disabled.
The helper returns complete tensors in the same layout at every TP size.
Capture also depends on the engine being able to discover the model's decoder
layers or MoE gates; test new model architectures before relying on their
activations.

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
| `budget_bytes` | Total limit (256 MiB by default) on raw CPU values and row labels, including pending transfers, across the stream's layers and workers. Exceeding it fails capture instead of returning a partial result. |
| `sample_indices` | Optional unique nonnegative input IDs, one per prompt. `capture_batches()` supplies the indices in its original prompt iterable. |
| `**generate_kwargs` | Forwarded into `SamplingParams` (e.g. `temperature`). |

Capture requires one continuation per prompt (`n=1`); multiple continuations
are rejected before contacting workers so sample ownership stays unambiguous.

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
result.sample_indices     # original input IDs; sample methods still use local indices
result.component          # captured stream, such as "hidden_states"
result.model              # model identifier when available
result.selection          # canonical global selection, or None for all rows
result.per_prompt_selections  # optional overrides; None entries use the global selection
result.to_nested()        # [sample][layer_pos] tensors, when explicitly needed
```

`result.meta(layer)` exposes the raw row labels (`req_ids` / `positions` / `token_ids`).
Labels are mandatory. `token()` indexes a sample's captured rows, so `-1`
means the last selected row, which may differ from the last original prompt
position. `sample_rows()` returns a view when those rows are contiguous;
noncontiguous rows are gathered in sequence order.
For bounded reads of a long sample, use
`result.iter_sample_rows(i, layer, chunk_size=32)`. Each yielded chunk contains
at most that many rows, including when the sample's rows are interleaved.

## Process a dataset in batches

Use `capture_batches()` with a prompt iterable. Batches respect both
`batch_size` and an estimate of selected-row storage. Text and token-ID inputs
use the same position-selection rules as the workers. Unknown multimodal
geometry is isolated into single-prompt batches, with worker admission still
enforcing the budget. A single prompt estimated above the budget is rejected
before generation. `sample_indices` records each sample's original input index;
methods such as `sample_rows(i, layer)` still use local indices within a batch.

The iterator starts the next capture only when the consumer asks for it. It
may read one prompt ahead to decide a byte-based batch boundary. Worker storage
is drained before yielding. Pass the iterator directly
to extraction to avoid retaining an entire corpus:

```python
from itertools import chain
from easysteer.extraction import extract

batches = hs.capture_batches(
    llm, chain(positive_prompts, negative_prompts),
    batch_size=32, layers=[10, 11, 12],
    select=SelectSpec(prompt_positions=[-1]),
)
labels = chain([True] * len(positive_prompts), [False] * len(negative_prompts))
vector = extract(batches, labels, method="diffmean")
vector.export_gguf("direction.gguf")
```

Labels are consumed in input order, one Boolean or integer 0/1 per sample.
`extract(..., token_pos="mean")` pools each sample's selected rows before averaging samples,
so longer prompts do not get extra weight. `method="incremental_pca"` provides
an explicit approximate PCA option with bounded working storage. See the
[extraction guide](extracting-vectors.md) for algorithms and memory preflight.

Both capture helpers default to these independent controls:

| Control | Default | Scope |
|---|---|---|
| `budget_bytes` | 256 MiB | Retained raw activation values and int32 labels, including pending transfers, across this stream's workers. |
| `device_budget_bytes` | 256 MiB | Per-worker fixed capture outputs and pending selection/cast data for this stream. |
| `staging_bytes` | 16 MiB | Reusable pinned transfer page per worker and stream. Stored datasets use pageable memory. |
| `fetch_bytes` | 16 MiB | Target raw size per fetch page across workers; at least one row is fetched. |

These are allocation controls, **not a total process RSS or GPU-memory cap**.
Model activations, KV cache, CUDA graph pools, allocator caches, Python labels,
RPC encoding/decoding copies and results retained by the caller add overhead.
Paging bounds each serialization/assembly temporary; the final in-memory
result still holds all selected rows of its batch. A single row can exceed the
fetch target. `None` explicitly disables a raw or device budget.

Attention raw budgets are divided equally across TP workers, including duplicate
row labels; replicated hidden states and router logits have one owner. The
`capture_status` RPC exposes per-layer row counts, storage, pending device data,
pinned staging allocation and graph memory. Raw-budget or device-budget
violations fail the result instead of silently returning incomplete data.

To keep final activation arrays on disk, provide a new output directory:

```python
result = hs.capture(llm, prompts, layers=[10], storage_dir="captures/run-001")
# Reopen later; arrays are memory-mapped and no pickle is loaded.
result = hs.CaptureResult.load("captures/run-001")
```

`capture_batches(..., storage_dir="captures/dataset")` writes one directory per
batch, named by its first global sample index. Iterate over those directories
and load one batch at a time for extraction. Supply fresh labels in saved
batch/sample order; extraction does not use `sample_indices` to index the labels:

```python
from pathlib import Path

def saved_batches():
    for path in sorted(Path("captures/dataset").glob("batch-*")):
        yield hs.CaptureResult.load(path)

vector = extract(saved_batches(), iter(dataset_labels), method="diffmean")
```

`dataset_labels` contains one label per saved sample in that order. A fresh
`saved_batches()` iterator can be used for another extraction call.
`result.save(path)` also persists an existing result. Saved captures retain the
global selection, per-prompt overrides and sample IDs. Loaded arrays use
copy-on-write mappings: editing a loaded tensor does not change the saved files.
Saved outputs retain request IDs, prompt/token information and
completion text/tokens, rather than every runtime field of `RequestOutput`.
Memory-mapped pages are managed by the operating system; touching every page
can increase resident memory. Keep input batches bounded even with disk storage.

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
`head_size` is the value-output dimension per head. Public results and layouts
use the model's global query-head order at every TP size. Neither KV head counts
nor the model's residual hidden size should be used to infer this layout.

This stream supports standard decoder MHA/GQA with ordinary tensor parallelism;
MLA, encoder attention, and cross-attention are not exposed as head outputs.
It shares capture's graph and prefix-cache policy. These activations feed the
[ITI extractor](extracting-vectors.md#iti-attention-head-directions).

## Multimodal inputs

Multimodal prompts use the same list of input dictionaries as `LLM.generate`,
including `prompt` and `multi_modal_data`; capture preserves their cache salts.

## Graph execution

Eligible FULL-graph batches use a separate CUDA graph that records activations.
This path supports ordinary tensor parallelism, requires no LoRA or speculative
decoding, and requires steering either disabled or running `in_graph`.
Ineligible capture batches, including piecewise-only execution, run eagerly;
steps with no selected rows keep ordinary model execution.
The attention backend must support FULL graphs for the batch being captured;
vLLM can downgrade a requested FULL configuration to decode-only execution.
The first eligible batch records the capture graph. One component/layer
combination stays cached across `capture()` calls; changing the selected token
positions, storage dtype or reduction does not require recording it again.
Changing components or layers replaces that cached graph. Ordinary model graphs
do not contain capture operations.

All TP ranks record and replay the same graph variant, including ranks that do
not retain replicated hidden states or router logits. Attention outputs stay
sharded during replay and are assembled when fetched. `capture_status` reports
`graph_replays` on every rank; a non-owner rank can have a ready graph with zero
capture buffer bytes.

The graph's fixed output buffers contain whole execution batches before row
selection. Selecting fewer rows reduces retained CPU data and transfer volume,
but does not shrink those fixed GPU buffers. Stopping capture clears the stream
while retaining the cached graph for reuse; status reports its buffer and total
allocation sizes separately. Before recording, capture checks discovered output
widths against the device budget and available GPU memory, including a reserve
based on the ordinary graph allocation. If these checks fail, every TP rank
uses eager capture. Unknown dimensions with a finite budget also use eager
capture. Graph-pool estimates are conservative admission checks, not a guarantee
against unrelated device allocations. Release cached capture graphs explicitly
when the workload is finished:

```python
hs.release_capture_cache(llm)
```

For a long-running TP engine that repeatedly changes captured components or
layers, set `disable_custom_all_reduce=True` when constructing `LLM`. vLLM's
custom all-reduce graph registration table has a fixed lifetime capacity;
replacing CUDA graphs does not reclaim those registrations. This option uses
vLLM's NCCL path and retains capture graph replay. Reusing the same capture
variant does not add registrations.

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
