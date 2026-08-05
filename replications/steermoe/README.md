# Steering MoE LLMs via Expert (De)Activation (SteerMoE)

[Paper Link](https://arxiv.org/abs/2509.09660) · [Official Code](https://github.com/adobe-research/SteerMoE)

## Abstract

Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token
through a subset of specialized Feed-Forward Networks (FFN), known as experts.
We present SteerMoE, a framework for steering MoE models by detecting and
controlling behavior-linked experts. Our detection method identifies experts
with distinct activation patterns across paired inputs exhibiting contrasting
behaviors. By selectively (de)activating such experts during inference, we
control behaviors like faithfulness and safety without retraining or modifying
weights.

## Method

1. **Detection**: run paired prompts showing opposite behaviors (e.g. counting
   with digits `1, 2, 3` vs. words `one, two, three`) through the model and
   record, for each expert at each layer, how often it lands in the router's
   top-k on the behavior-relevant tokens. The **risk difference**
   `Δ(layer, expert) = p_behavior1 − p_behavior2` ranks experts by behavior
   association.
2. **Steering**: at inference, log-softmax the router logits, then force
   *activated* experts to the per-token max score + ε and *deactivated*
   experts to the per-token min score − ε, before top-k expert selection.
   Untouched experts keep their relative weights, so the mixture stays intact.

## Replication with EasySteer

The official implementation forks one vLLM model file per architecture (a
`modeling_vllm/` copy for steering, a `modeling_vllm_save/` copy for capturing
router logits). EasySteer replicates both phases with **no model-code changes,
on any FusedMoE architecture**:

| SteerMoE phase | Official implementation | EasySteer |
| --- | --- | --- |
| Record router logits | per-model forked file writing `.npy`s | `router_logits` capture stream (gate forward hooks) |
| Detect experts | offline risk difference | same, from the captured stream |
| Steer experts | per-model forked forward | `moe_router` algorithm, `activate`/`deactivate` modes |

`steermoe.ipynb` runs both phases in one engine (replicating
`custom_steering.ipynb` from the official repo):

- **Detection** — capture per-token router logits for a few contrastive
  digits-vs-words pairs, compute the risk difference, and save the 200
  most word-linked experts as `steermoe_qwen3_words.json`.
- **Steering** — deactivate those experts, flipping greedy counting from
  written words to digits, and verify the mechanism by re-capturing
  router logits *post-steering* (deactivated experts disappear from
  every token's top-8).

Model: [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B)
(48 MoE layers × 128 experts, top-8), one of the models evaluated in the
paper.

A steering config is a JSON file mapping layers to expert lists:

```json
{
  "layer_configs": {
    "7":  {"mode": "deactivate", "expert_ids": [30, 25]},
    "9":  {"mode": "activate", "activate_ids": [12], "deactivate_ids": [35]}
  }
}
```

applied per request via:

```python
steering = SteeringSpec(vectors=[
    VectorSpec(
        source="steermoe_qwen3_words.json",
        algorithm="moe_router",
        layers=[...],  # the layers present in the JSON
        apply=ApplySpec(phases=["prompt", "generation"]),
    ),
])
llm.generate(prompts, params, steering=steering)
```

Note: models whose MoE blocks fuse the gate weights into the MoE runner
(shared-expert architectures like Qwen1.5/2-MoE) bypass the gate module
forward, so neither router-logit capture nor gate steering can hook them;
EasySteer logs a warning for such blocks. OLMoE, Qwen3-MoE, Mixtral, and
GPT-OSS use the hookable path.
