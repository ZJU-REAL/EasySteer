# OpenAI-compatible server

EasySteer serves steering over vLLM's standard OpenAI-compatible HTTP API.

## Start the server

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-steer-vector --port 8017 --enforce-eager
```

## Per-request steering

Pass the `SteeringSpec` as JSON in the `steering` field — via `extra_body` with the
OpenAI SDK, or directly with `curl`:

=== "Python (OpenAI SDK)"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8017/v1", api_key="EMPTY")

    steering = {
        "vectors": [{
            "source": "vectors/happy_diffmean.gguf",
            "scale": 2.0,
            "layers": list(range(10, 26)),
            "normalize": True,
            "apply": {"phases": ["prompt", "generation"]},
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
      -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": "Comfort Alice about her dog."}],
        "max_tokens": 128,
        "steering": {
          "vectors": [{
            "source": "vectors/happy_diffmean.gguf",
            "scale": 2.0,
            "layers": [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
            "apply": {"phases": ["prompt", "generation"]}
          }]
        }
      }'
    ```

The JSON shape mirrors the Python spec exactly (`vectors` / `conflict` / `debug`, each
vector with `source`, `algorithm`, `scale`, `layers`, `normalize`, `apply`, `params`).

## Server-level (engine default) steering

Apply one spec to every request:

```bash
vllm serve <model> --enable-steer-vector --steering-config spec.json
```

Replace it at runtime (this resets the prefix cache):

```bash
curl -X POST http://localhost:8017/v1/steering \
  -H "Content-Type: application/json" \
  -d '{"spec": {"vectors": [...]}}'
```

Server-level and per-request steering cannot be combined in one request.

<!-- TODO: document the /v1/steering GET/DELETE surface (if present) and auth story. -->
