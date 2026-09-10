# Engine configuration

Steering has two configuration levels. Engine arguments declare the algorithms,
execution mode, and capacity for the lifetime of an `LLM` or `vllm serve`
process. Each request supplies its vectors and token selection through a
[`SteeringSpec`](steering-specs.md).

## Python and CLI arguments

Pass Python arguments to `LLM(...)`; use the corresponding flags with
`vllm serve MODEL ...`.

| Python argument | CLI flag | Default | Purpose |
|---|---|---|---|
| `enable_steer_vector` | `--enable-steer-vector` | `False` | Enable steering. A startup `steering_config` also enables it. |
| `steer_algorithms` | `--steer-algorithms` | `None` | Algorithm list, comma-separated names, or `"all"`. Required when enabling steering without a startup default. |
| `steer_multi_vector` | `--steer-multi-vector` | `False` | Allow more than one vector in a request. Implied by `"all"` or a multi-vector startup default. |
| `steer_graph_mode` | `--steer-graph-mode` | `"auto"` | Select `in_graph` or `split`; see [graph modes](../user-guide/performance.md#graph-modes). |
| `steer_graph_max_rank` | `--steer-graph-max-rank` | `32` | Maximum LoReFT/LM-Steer rank accepted by an `in_graph` engine. |
| `max_steer_vectors` | `--max-steer-vectors` | `min(256, max_num_seqs)` | Maximum distinct steering configurations active in a running batch. |
| `steer_vector_dtype` | `--steer-vector-dtype` | `"auto"` | Payload dtype: model dtype for `auto`, or `float16`, `bfloat16`, `float32`. |
| `steer_require_preload` | `--steer-require-preload` | `False` | Require file-backed payloads to be preloaded before generation requests. |
| `steering_config` | `--steering-config` | `None` | Startup default as JSON text or the path to a spec JSON file. |

The declaration can include several algorithms while each request uses just one.
For example, `steer_algorithms=["direct", "attention_add"]` declares two
algorithms, while `steer_multi_vector=True` permits combining vectors in a
single spec. These are separate choices.

When a startup default is supplied, its algorithms are added to the declaration.
The engine can inspect that default's payload to choose its graph mode. Later
default updates and per-request overrides must fit the resolved engine settings.

## Common configurations

=== "One additive intervention per request"

    ```python
    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        enable_steer_vector=True,
        steer_algorithms=["direct"],
    )
    ```

    Auto selects `in_graph`. Requests can use different files, scales, layers,
    selectors, and normalization settings within that declaration.

=== "LoReFT with a known rank bound"

    ```python
    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        enable_steer_vector=True,
        steer_algorithms=["loreft"],
        steer_graph_mode="in_graph",
        steer_graph_max_rank=32,
    )
    ```

    Each request must supply a LoReFT payload of rank at most 32. Use `auto`
    to select `split` when the declaration must accept arbitrary ranks.

=== "Interactive multi-algorithm server"

    ```bash
    vllm serve Qwen/Qwen2.5-1.5B-Instruct \
      --enable-steer-vector --steer-algorithms all \
      --max-steer-vectors 16 --port 8017
    ```

    `all` implies multi-vector support and selects `split`. Component-specific
    requests still require a compatible model; declaring `moe_router`, for
    example, does not add MoE gates to a dense model.

## Capacity and defaults

`max_steer_vectors` counts active **configurations**, not the number of weight
files or submitted requests. Requests with identical configurations share a
slot. If all slots are in use, requests needing a new configuration wait for
one to become available. This does not require the client to resubmit them.

Payload storage is content-addressed separately: several configurations that
use the same weights can share a materialized payload. A larger slot capacity
permits more distinct concurrent configurations, but increases persistent
graph-table memory. Low-rank graph buffers also grow with
`steer_graph_max_rank`; use a bound suited to the checkpoints you serve.

Changing the engine default with `llm.set_default_steering(...)` affects new
requests. It does not reserve a slot permanently or change the graph mode.
See [request inheritance](../user-guide/steering.md#attaching-a-spec) and
[HTTP management endpoints](../user-guide/openai-server.md#management-endpoints).

## Standard vLLM settings

Model loading, scheduling, sampling, memory allocation, and CUDA graph capture
sizes remain vLLM settings. The fork uses vLLM 0.29.0 and its V2 GPU runner for
steering and capture; leave the runner selection at its default.

Use the upstream [engine argument reference](https://docs.vllm.ai/en/latest/configuration/engine_args/)
for general options and this page for EasySteer's additions. Match upstream
documentation to the installed vLLM version when defaults change.
