# Paper replications

The [`replications/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications)
directory implements published steering methods with EasySteer notebooks. The folders
contain paper notes, notebooks, and available vector or intervention artifacts.
Consult each experiment for its required model, data and checkpoint files.

| Folder | One-liner | Category | Component |
|---|---|---|---|
| [`bipo/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/bipo) | Bi-directional preference optimization vectors steering power-seeking behavior | Personalization | `hidden_states` |
| [`cast/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/cast) | Conditional activation steering to program refusal (CAST) | Safety | `hidden_states` |
| [`controlingthinkingspeed/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/controlingthinkingspeed) | Speeding up / slowing down reasoning-model thinking on MATH500 | Reasoning | `hidden_states` |
| [`creative_writing/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/creative_writing) | Steering LLMs to evaluate and amplify creativity | Style | `hidden_states` |
| [`fractreason/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/fractreason) | Fractional reasoning via latent steering vectors for inference-time compute | Reasoning | `hidden_states` |
| [`improve_reasoning/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/improve_reasoning) | Representation-engineering vectors that improve reasoning performance | Reasoning | `hidden_states` |
| [`iti/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/iti) | Llama-2-7B-Chat attention head steering with a TruthfulQA accuracy comparison and examples | Truthfulness | `attention_heads` |
| [`lm_steer/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/lm_steer) | Word embeddings as steers for language models (LM-Steer, GPT-2) | General | `hidden_states` |
| [`loreft/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/loreft) | ReFT: train and apply LoReFT representation finetuning | General | `hidden_states` |
| [`refusal_direction/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/refusal_direction) | Refusal is mediated by a single direction (DiffMean ablation) | Safety | `hidden_states` |
| [`sae_entities/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/sae_entities) | SAE entity-knowledge directions and hallucination awareness | Truthfulness | `hidden_states` |
| [`sake/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/sake) | SAKE: steering activations for knowledge editing | Knowledge | `hidden_states` |
| [`seal/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/seal) | SEAL: steerable reasoning calibration (execution/reflection/transition vectors) | Reasoning | `hidden_states` |
| [`sharp/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/sharp) (VLM) | SHARP: steering hallucination in LVLMs via representation engineering (EMNLP 2025) | Truthfulness | `hidden_states` |
| [`steerable_chatbot/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/steerable_chatbot) | Personalizing LLMs with preference-based activation steering | Style | `hidden_states` |
| [`steermoe/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/steermoe) | SteerMoE: expert (de)activation steering of MoE routers on Qwen3-30B-A3B (arXiv:2509.09660) | MoE | `router_logits` |

Component names match the API: `hidden_states` denotes decoder block output,
`attention_heads` denotes head outputs before the attention output projection,
and `router_logits` denotes MoE routing scores.

Contributions of new replications are welcome — see
[Contributing](../developer-guide/contributing.md).

<!-- TODO: link each row to the paper (arXiv) and note which notebooks need which
models/GPUs. -->
