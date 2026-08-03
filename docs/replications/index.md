# Paper replications

The [`replications/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications)
directory reproduces published steering papers with EasySteer notebooks. Each folder
contains a README, the notebook(s), and the extracted vectors.

| Folder | One-liner | Category |
|---|---|---|
| [`bipo/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/bipo) | Bi-directional preference optimization vectors steering power-seeking behavior | Personalization |
| [`cast/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/cast) | Conditional activation steering to program refusal (CAST) | Safety |
| [`controlingthinkingspeed/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/controlingthinkingspeed) | Speeding up / slowing down reasoning-model thinking on MATH500 | Reasoning |
| [`creative_writing/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/creative_writing) | Steering LLMs to evaluate and amplify creativity | Style |
| [`fractreason/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/fractreason) | Fractional reasoning via latent steering vectors for inference-time compute | Reasoning |
| [`improve_reasoning/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/improve_reasoning) | Representation-engineering vectors that improve reasoning performance | Reasoning |
| [`lm_steer/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/lm_steer) | Word embeddings as steers for language models (LM-Steer, GPT-2) | General |
| [`loreft/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/loreft) | ReFT: train and apply LoReFT representation finetuning | General |
| [`refusal_direction/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/refusal_direction) | Refusal is mediated by a single direction (DiffMean ablation) | Safety |
| [`sae_entities/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/sae_entities) | SAE entity-knowledge directions and hallucination awareness | Reality |
| [`sake/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/sake) | SAKE: steering activations for knowledge editing | Knowledge |
| [`seal/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/seal) | SEAL: steerable reasoning calibration (execution/reflection/transition vectors) | Reasoning |
| [`sharp/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/sharp) | SHARP: steering hallucination in LVLMs via representation engineering (EMNLP 2025) | Reality |
| [`steerable_chatbot/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/steerable_chatbot) | Personalizing LLMs with preference-based activation steering | Style |
| [`steermoe/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/steermoe) | SteerMoE: expert (de)activation steering of MoE routers on OLMoE-1B-7B (arXiv:2509.09660) | MoE |

Contributions of new replications are welcome — see
[Contributing](../developer-guide/contributing.md).

<!-- TODO: link each row to the paper (arXiv) and note which notebooks need which
models/GPUs. -->
