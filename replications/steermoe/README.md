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

- `expert_detection.ipynb` — capture per-token router logits for a few
  contrastive pairs, compute the risk difference, and write a `steermoe`
  layer config (replicates `custom_steering.ipynb` from the official repo).
- `steermoe_steer.ipynb` — steer with the detected experts (deactivating
  100 digit-linked experts flips greedy counting to written number words),
  verify the mechanism by re-capturing router logits *post-steering*
  (deactivated experts disappear from every token's top-8), and steer
  faithfulness both ways with the paper's released expert rankings for
  OLMoE-1B-7B (downloaded from the official repo; Adobe Research License,
  noncommercial research).

Model: [`allenai/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct)
(16 MoE layers × 64 experts, top-8), one of the six models evaluated in the
paper. The expert-ranking pickle
`activations_[allenai--OLMoE-1B-7B-0125-Instruct]_[faithfulness].pkl` comes
from the official repo.

A steering config is a JSON file mapping layers to expert lists:

```json
{
  "layer_configs": {
    "7":  {"mode": "deactivate", "expert_ids": [30, 25]},
    "9":  {"mode": "activate", "activate_ids": [12], "deactivate_ids": [35]}
  }
}
```

(`boost`, `suppress`, `soft_hard` and `steermoe` are accepted as
deprecated aliases of `activate`/`deactivate`.)

applied per request via:

```python
SteerVectorRequest(
    "steer-away-from-digits", 1,
    steer_vector_local_path="steermoe_digits.json",
    algorithm="moe_router",
    prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1],
)
```

Note: models whose MoE blocks fuse the gate weights into the MoE runner
(shared-expert architectures like Qwen1.5/2-MoE) bypass the gate module
forward, so neither router-logit capture nor gate steering can hook them;
EasySteer logs a warning for such blocks. OLMoE, Qwen3-MoE, Mixtral, and
GPT-OSS use the hookable path.
