# Quickstart

Steer a chat model toward a "happy" direction and compare against the baseline.

## 1. Start a steering-enabled engine

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enable_steer_vector=True,
    steer_algorithms=["direct"],
    tensor_parallel_size=1,
)
```

`steer_algorithms` declares what requests may use. For this single-vector
`direct` workload, the default `steer_graph_mode="auto"` selects `in_graph`.
Leave compilation, chunked prefill, and prefix caching at their engine defaults.
The [performance guide](../user-guide/performance.md) explains how these settings
interact with steering.

If needed, select a GPU with `CUDA_VISIBLE_DEVICES` before starting Python. Run
from the EasySteer repository root so the bundled vector path resolves.

## 2. Describe the steering with a spec

A steering configuration is three nested objects — see the
[Steering guide](../user-guide/steering.md) for the full language:

```python
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

def happy_steering(scale):
    return SteeringSpec(vectors=[VectorSpec(
        source="vectors/happy_diffmean.gguf",  # vector file (GGUF)
        scale=scale,                            # strength; 0.0 = no effect
        layers=list(range(10, 24)),             # layers to steer
        apply=ApplySpec(prompt="all", generation="all"),
    )])
```

## 3. Generate with and without steering

```python
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "Alice's dog has passed away. Please comfort her."},
]
prompt = {"prompt_token_ids": llm.get_tokenizer().apply_chat_template(
    messages, tokenize=True, return_dict=False, add_generation_prompt=True,
)}
sampling_params = SamplingParams(
    temperature=0.0, max_tokens=128, repetition_penalty=1.1,
)

baseline = llm.generate(prompt, steering=False, sampling_params=sampling_params)
happy = llm.generate(
    prompt, steering=happy_steering(2.0), sampling_params=sampling_params,
)

print(baseline[0].outputs[0].text)  # ordinary condolences
print(happy[0].outputs[0].text)     # conspicuously upbeat
```

The tokenizer supplies the model's chat format. `steering=False` explicitly
disables steering, including an engine default. Positive and negative scales
move in opposite directions; the bundled vector and selected layers are for
this Qwen model. Use a vector extracted for the target model when changing it.

## Where the vector came from

`happy_diffmean.gguf` was produced by capturing hidden states on contrastive prompts and
taking the difference of means — the full pipeline is:

1. [Capture hidden states](../user-guide/hidden-state-capture.md) with
   `easysteer.hidden_states.capture()`.
2. [Extract a vector](../user-guide/extracting-vectors.md) with
   `easysteer.steer.extract_diffmean_control_vector()` and export it as GGUF.
3. Apply it at inference with a `SteeringSpec` (this page).

## Next steps

- Serve steering over HTTP: [OpenAI-compatible server](../user-guide/openai-server.md)
- Experiment without code: [Web demo](../user-guide/web-demo.md)
- Browse [paper replications](../replications/index.md) for end-to-end worked examples.
