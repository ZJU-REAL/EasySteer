# OpenAI-compatible server

Use vLLM's OpenAI-compatible HTTP API to serve baseline and steered requests
from one engine. The request's `steering` field uses the same
[spec schema](../api-reference/steering-specs.md) as Python inference.

## Requirements

Install the [EasySteer fork](../getting-started/installation.md) on the server.
Clients need only an HTTP client or the OpenAI SDK. A vector's `source` path
must be available to the server; use a JSON payload in `data` when sending
weights from the client.

The examples below use an unauthenticated local server. If the server sets
`--api-key`, supply that key to the SDK and include `Authorization: Bearer <key>`
in curl requests, including management requests.

## Start the server

Run from the EasySteer repository root so the server can resolve the examples'
relative `vectors/` paths:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --enable-steer-vector --steer-algorithms direct \
  --port 8017
```

The default graph mode is selected from the declared algorithms: this
single-vector `direct` server uses `in_graph`. Multi-vector requests also
require `--steer-multi-vector`, which selects `split` while retaining CUDA
graphs. For other algorithms and capacity settings, see
[engine configuration](../api-reference/engine-configuration.md).

Wait until the server reports that it is ready. `GET /v1/models` lists the model
IDs accepted in requests; use one of those IDs in the client's `model` field.

## Send a steering request

Pass the `SteeringSpec` as JSON in the `steering` field, using `extra_body` with
the OpenAI SDK or sending the field directly with `curl`. Both examples use the
README's happy-vector and Alice prompt:

=== "Python (OpenAI SDK)"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8017/v1", api_key="EMPTY")
    steering = {
        "vectors": [{
            "source": "vectors/happy_diffmean.gguf",
            "algorithm": "direct",
            "scale": 2.0,
            "layers": list(range(10, 26)),
            "apply": {"prompt": "all", "generation": "all"},
        }]
    }
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{"role": "user", "content": "Alice's dog has passed away. Please comfort her."}],
        max_tokens=128,
        temperature=0.0,
        extra_body={"steering": steering},
    )
    print(response.choices[0].message.content)
    ```

=== "curl"

    ```bash
    curl http://localhost:8017/v1/chat/completions \
      -H "Content-Type: application/json" \
      --data-binary @- <<'JSON'
    {
      "model": "Qwen/Qwen2.5-1.5B-Instruct",
      "messages": [{"role": "user", "content": "Alice's dog has passed away. Please comfort her."}],
      "max_tokens": 128,
      "temperature": 0.0,
      "steering": {
        "vectors": [{
          "source": "vectors/happy_diffmean.gguf",
          "algorithm": "direct",
          "scale": 2.0,
          "layers": [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
          "apply": {"prompt": "all", "generation": "all"}
        }]
      }
    }
    JSON
    ```

Use `extra_body={"steering": False}` (JSON: `"steering": false`) for the baseline
comparison. `normalize` defaults to `false` in both examples.

The JSON shape mirrors the Python spec (`vectors` / `conflict`, each vector with
`source` or `data`, `algorithm`, `scale`, `layers`, `normalize`, `apply`, `params`,
and `name`). A `source` is a path on the server. For an in-memory payload,
base64-encode the tensor bytes before sending its wire dictionary as a vector's
`data` field:

```python
from easysteer.vectors import from_gguf, to_json_payload

payload = from_gguf("vectors/happy_diffmean.gguf")
wire = to_json_payload(payload)
# Use {"data": wire, "algorithm": "direct", "apply": {...}} in vectors.
```

The [payload reference](../api-reference/algorithms.md) lists all supported
native formats and third-party adapters. Startup options and CLI equivalents
are listed in [engine configuration](../api-reference/engine-configuration.md).

## Set a startup default

Save a complete spec as `spec.json`, for example:

```json
{
  "vectors": [{
    "source": "vectors/happy_diffmean.gguf",
    "algorithm": "direct",
    "scale": 2.0,
    "layers": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "apply": {"prompt": "all", "generation": "all"}
  }]
}
```

Start an engine that uses it when a request omits steering. The startup spec
supplies the algorithm declaration:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --steering-config spec.json --port 8017
```

An omitted or `null` `steering` field inherits the default. A request spec
overrides it completely; `"steering": false` explicitly disables steering.
For example, use `extra_body={"steering": False}` for an unsteered comparison
with the OpenAI Python client.

## Management endpoints

Use these endpoints to update a default or preload vectors while the model stays
loaded. An engine started without a default can set one later if steering is
enabled and the algorithms are declared.

!!! note "Runtime default updates"
    Changing the default at runtime requires one API frontend
    (`--api-server-count=1`). Startup defaults and explicit per-request steering
    also work with multiple API frontends. Runtime updates are rejected there.

| Endpoint | Request / result |
|---|---|
| `GET /v1/steering` | Returns `{"active": false}` when no default is configured, otherwise its status and authoring spec. In-memory tensor data is omitted and summarized by payload kind and content hash. |
| `POST /v1/steering` | Sets or replaces the default with `{"spec": ...}`; `{"spec": null}` clears it. Requires steering to be enabled, with the requested algorithms declared. |
| `GET /v1/steering/vectors` | Lists preloaded paths as `{"preloaded": [...]}`. |
| `POST /v1/steering/vectors` | Preloads `{"paths": [...], "algorithm": "direct", "params": {...}}`. `params` is optional. Requires steering to be enabled. |

### Replace or clear the default

Replace the engine default from a revised `spec.json`:

```bash
python -c 'import json; print(json.dumps({"spec": json.load(open("spec.json"))}))' \
  | curl -X POST http://localhost:8017/v1/steering \
      -H "Content-Type: application/json" --data-binary @-
```

Updates must stay within the engine's declared algorithms, multi-vector support,
and graph capabilities. Invalid specs or malformed management request bodies
return HTTP 400; worker failures remain server errors.
Updates apply only to new requests; admitted requests retain their configuration
and weight snapshot. Prefix-cache entries remain separated by the effective
configuration, so updates do not require a cache reset.

Clear the default without restarting the server:

```bash
curl -X POST http://localhost:8017/v1/steering \
  -H 'Content-Type: application/json' -d '{"spec": null}'
```

### Preload file-backed vectors

`paths` must be a non-empty list of non-empty strings. `params`, if supplied,
must be an object or `null`; malformed requests return HTTP 400.

To require advance loading of file-backed vectors, start the server with
`--steer-require-preload`, then preload the file before sending generation requests:

```bash
curl -X POST http://localhost:8017/v1/steering/vectors \
  -H "Content-Type: application/json" \
  -d '{"paths": ["vectors/happy_diffmean.gguf"], "algorithm": "direct"}'
```

Preloading records the algorithm and content snapshot. After changing a source
file, preload it again before using the new contents with
`--steer-require-preload`. Router parameters that change the payload must match
between preloading and the request. For example, preload a router file with
`{"paths": ["router.json"], "algorithm": "moe_router", "params": {"mode": "soft", "lambda": 0.7}}`
before using that file with the same `VectorSpec.params`. Changing only the
request's scale, layer subset or selection reuses the preloaded payload.

The Python equivalent is
`llm.preload_steer_vectors(paths, algorithm="moe_router", params={"mode": "soft", "lambda": 0.7})`.
