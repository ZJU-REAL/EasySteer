# Hidden-state capture

`easysteer.hidden_states` extracts intermediate activations from a running vLLM
engine for analysis, training data and steering-vector extraction. Capture works
without enabling steering.

## Engine requirements

Capture requires the V2 model runner; V1 is not supported. In vLLM 0.28,
ordinary dense models use V2 by default, including the Qwen model below.
For architectures that do not select V2 by default, set
`VLLM_USE_V2_MODEL_RUNNER=1` before importing vLLM.

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

Eligible FULL-graph batches use a separate CUDA graph that records activations.
This path requires one worker, no LoRA or speculative decoding, and steering
either disabled or running `in_graph`. Other batches that need captured rows
run eagerly; steps with no selected rows keep ordinary model execution.
The first eligible batch records the capture graph. One component/layer
combination stays cached across `capture()` calls; changing the selected token
positions, storage dtype or reduction does not require recording it again.
Changing components or layers replaces that cached graph. Ordinary model graphs
do not contain capture operations.

## `hs.capture()`

One call captures a batch of prompts and returns a labelled
[`CaptureResult`](../api-reference/hidden-states.md):

```python
prompts = ["What is steering?", "Explain PCA."]
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
| `stream` | `"hidden_states"` (default) or `"router_logits"` (MoE). |
| `steering` | `SteeringSpec` or per-prompt list, as in `LLM.generate()`. Captured values include steering. |
| `**generate_kwargs` | Forwarded into `SamplingParams` (e.g. `temperature`). |

### Select clauses

Capture's `SelectSpec` and steering's `ApplySpec` share one selection language —
per-phase `"all"` (`prompt="all"` / `generation="all"`), six phase-scoped include
selectors (prompt tokens/positions/window, generation tokens/positions/window)
and their symmetric exclude twins — resolved identically by the engine, so a
clause means the same thing in both systems:

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

## Working with `CaptureResult`

Rows are grouped by their owning request via engine labels and ordered by sequence
position — the only correct grouping under continuous batching.

```python
result.layer_ids          # sorted true layer ids
result.rows(12)           # Tensor(total_rows, dim) for layer 12, all samples
result.sample(0)          # {layer_id: Tensor(rows, dim)} for sample 0
result.sample_positions(0)  # absolute sequence positions of sample 0's rows
result.sample_token_ids(0)  # input token ids of sample 0's rows
result.outputs            # the vLLM RequestOutput list, prompt order
result.to_nested()        # legacy shape: [sample][layer_pos] tensors
```

`result.meta(layer)` exposes the raw row labels (`req_ids` / `positions` / `token_ids`);
`result.labelled` tells you whether per-sample views are available.

## MoE router logits

Pass `stream="router_logits"` to `hs.capture()` on an MoE model to capture per-token
router logits instead of hidden states (used e.g. by the
[SteerMoE replication](../replications/index.md)).

## Compatibility wrappers

`get_all_hidden_states_generate` and `get_moe_router_logits_generate` are thin
wrappers over `capture()`. The hidden-state wrapper returns
`(hidden_states, outputs)`, with nested `[sample][layer]` tensors by default or
concatenated `[layer]` tensors when `split_by_samples=False`. The router-logit
wrapper returns `(router_logits, outputs)`, with a `{layer_id: tensor}` dictionary
by default or one such dictionary per sample when `split_by_samples=True`.

Prefer `capture()` for new code: it preserves true layer IDs and per-sample
metadata. In particular, pass the `CaptureResult` directly to the vector
extractors when capturing a layer subset; converting to a plain nested list
loses the mapping from list positions to true layer IDs. The embed-task variants
(`get_all_hidden_states`, `get_moe_router_logits`) and the `vllm.hidden_states`
alias package were removed.

Multimodal prompts use the same list of input dictionaries as `LLM.generate`,
including `prompt` and `multi_modal_data`; capture preserves their cache salts.

## Prefix caching

KV blocks do not contain the hidden states or router logits skipped by a prefix
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
