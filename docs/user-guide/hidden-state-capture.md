# Hidden-state capture

`easysteer.hidden_states` extracts intermediate activations from a running vLLM engine —
the raw material for steering vectors.

## Engine requirements

```python
from vllm import LLM
import easysteer.hidden_states as hs

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
# Capture works on any engine configuration: capture-active batches
# dispatch through the eager forward automatically, and capture
# requests are cache-salted so prefix-cache hits never skip the
# recomputation capture needs.
```

## `hs.capture()`

One call captures a batch of prompts and returns a labelled
[`CaptureResult`](../api-reference/hidden-states.md):

```python
result = hs.capture(
    llm,
    prompts=["What is steering?", "Explain PCA."],
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
| `dtype` | Engine-side storage dtype, e.g. `"float16"`. |
| `select` | Global `SelectSpec` (or wire dict) choosing which rows to keep. |
| `per_prompt_selects` | One `SelectSpec` per prompt, overriding the global selection (`None` entries keep the global one). |
| `stream` | `"hidden_states"` (default) or `"router_logits"` (MoE). |
| `**generate_kwargs` | Forwarded into `SamplingParams` (e.g. `temperature`). |

### Select clauses

Row selection reuses the same `SelectSpec` language as steering's `ApplySpec` — a
phase gate, six phase-scoped include selectors (prompt tokens/positions/window,
generation tokens/positions/window) and their symmetric exclude twins — resolved identically by the
engine, so a clause means the same thing in both systems:

```python
from vllm.steer_vectors import SelectSpec

# Keep only the last prompt token of each sample
result = hs.capture(llm, prompts,
                    select=SelectSpec(phases=["prompt"], prompt_positions=[-1]))
```

`per_prompt_selects` requires `positions='all'` semantics (no reductions).

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

`get_all_hidden_states_generate` and `get_moe_router_logits_generate` predate
`capture()` and keep their original nested-list signatures, but are now thin
wrappers over it — splitting is always exact and label-driven. Prefer
`capture()` for new code; it additionally exposes select clauses, layer
subsets and per-sample metadata. The embed-task variants
(`get_all_hidden_states`, `get_moe_router_logits`) and the `vllm.hidden_states`
alias package were removed.

<!-- TODO: document capture of multimodal (VLM) prompts and streaming/chunked capture
sessions in more depth. -->
