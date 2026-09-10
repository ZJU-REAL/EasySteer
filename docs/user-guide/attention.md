# Attention head intervention

`attention_add` adds a direction to attention head outputs after attention
aggregation and before the output projection. This is the intervention point
used by [Inference-Time Intervention (ITI)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html).
It changes the representation passed to the output projection; it does not edit
attention scores, attention probabilities, or stored keys and values directly.

## Supported component

Use standard decoder MHA or GQA with `tensor_parallel_size=1`. MLA, encoder
attention, and cross-attention are outside this component's current scope.
The engine discovers attention modules from the model itself and checks that
the hook target is available.

For a layer with `H` query heads and value-output dimension `D`, the captured
and steered tensor has shape `(rows, H * D)`. This width need not equal the
residual hidden size. GQA still uses the **query** head count at this point,
not the number of KV heads.

## Capture the representation

The same capture API and selectors work for attention heads:

```python
from vllm import LLM
from vllm.capture import SelectSpec
import easysteer.hidden_states as hs

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    tensor_parallel_size=1,
    enable_steer_vector=True,
    steer_algorithms=["attention_add"],
)
prompts = ["The capital of France is"]
heads = hs.capture(
    llm,
    prompts,
    layers=[10],
    stream="attention_heads",
    select=SelectSpec(prompt_positions=[-1]),
    max_tokens=1,
    steering=False,
)
layout = heads.layouts[10]
last_token_heads = heads.sample(0)[10].reshape(
    -1, layout["num_heads"], layout["head_size"],
)
```

This raw completion prompt illustrates the component layout. For chat
applications, render the model's chat template as in the
[quickstart](../getting-started/quickstart.md). For a paper replication, keep
the paper's prompt format.

## Apply selected head directions

Use a concatenated `DirectionVector` or an EasySteer direction GGUF. Leave
unselected head slices zero. The same per-request `ApplySpec`, prefix caching,
and graph rules apply:

```python
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

# This file must contain attention directions for the model loaded above.
attention = SteeringSpec(vectors=[VectorSpec(
    source="vectors/my_attention_direction.gguf",
    algorithm="attention_add",
    scale=3.0,
    normalize=False,
    apply=ApplySpec(prompt_positions=[-1], generation="all"),
)])
outputs = llm.generate(prompts, steering=attention)
```

The last prompt position computes the first answer-token prediction; including
it is useful when steering a whole answer. `generation="all"` covers subsequent
decode forward passes. To steer all prompt processing as well, use
`ApplySpec(prompt="all", generation="all")`.

`attention_add` supports eager, `split`, and `in_graph`; a single-vector
declaration selects `in_graph` under the default `auto` mode. Normalization is
not supported, so keep `normalize=False`.

## ITI example

The [ITI notebook](https://github.com/ZJU-REAL/EasySteer/blob/main/replications/iti/iti.ipynb)
uses Llama-2-7B-Chat, the authors' QA prompt, 48 selected heads, and intervention
strength 15. It loads a supplied direction, evaluates one fixed TruthfulQA set,
and displays baseline/steered accuracy and generated examples as executed
notebook outputs.

To learn directions from your own data, use
[`ITIExtractor`](extracting-vectors.md#iti-attention-head-directions). It fits
per-head probes on training captures, selects heads by validation accuracy, and
returns the scaled head directions in the common control-vector format.
Choose held-out data by question before collecting answer representations.

The bundled replication direction uses the official implementation's separate
GEN calibration bank for the projection standard deviation. The general
extractor uses the provided training and validation captures. Both use
`attention_add` for inference; the extraction guide describes the distinction.
