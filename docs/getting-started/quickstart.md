# Quickstart

Steer a chat model toward a "happy" direction and compare against the baseline.

## 1. Start a steering-enabled engine

```python
import os
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# enable_steer_vector=True turns on steering support; without it the
# engine behaves like stock vLLM. steer_algorithms declares the
# algorithms requests will use — the engine derives the fastest
# CUDA-graph integration that serves them (undeclared algorithms are
# rejected; declare "all" to allow everything).
llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enable_steer_vector=True,
    steer_algorithms=["direct"],
    tensor_parallel_size=1,
)
```

## 2. Describe the steering with a spec

A steering configuration is three nested objects — see the
[Steering guide](../user-guide/steering.md) for the full language:

```python
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

def happy_steering(scale):
    return SteeringSpec(vectors=[VectorSpec(
        source="vectors/happy_diffmean.gguf",  # vector file (GGUF)
        scale=scale,                            # strength; 0.0 = no effect
        layers=list(range(10, 26)),             # layers to steer
        apply=ApplySpec(phases=["prompt", "generation"]),
    )])
```

## 3. Generate with and without steering

```python
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
text = ("<|im_start|>user\nAlice's dog has passed away. Please comfort her."
        "<|im_end|>\n<|im_start|>assistant\n")

baseline = llm.generate(text, steering=happy_steering(0.0),
                        sampling_params=sampling_params)
happy = llm.generate(text, steering=happy_steering(2.0),
                     sampling_params=sampling_params)

print(baseline[0].outputs[0].text)  # ordinary condolences
print(happy[0].outputs[0].text)     # conspicuously upbeat
```

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
