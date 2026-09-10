# Paper replications

The [`replications/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications)
directory implements published steering methods with EasySteer notebooks. The folders
contain paper notes, notebooks, and available vector or intervention artifacts.
Consult each experiment for its required model, data and checkpoint files.

| Folder | One-liner | Category | Component |
|---|---|---|---|
| [`bipo/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/bipo) | Bi-directional preference optimization vectors steering power-seeking behavior | Personalization | Residual stream |
| [`cast/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/cast) | Conditional activation steering to program refusal (CAST) | Safety | Residual stream |
| [`controlingthinkingspeed/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/controlingthinkingspeed) | Speeding up / slowing down reasoning-model thinking on MATH500 | Reasoning | Residual stream |
| [`creative_writing/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/creative_writing) | Steering LLMs to evaluate and amplify creativity | Style | Residual stream |
| [`fractreason/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/fractreason) | Fractional reasoning via latent steering vectors for inference-time compute | Reasoning | Residual stream |
| [`improve_reasoning/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/improve_reasoning) | Representation-engineering vectors that improve reasoning performance | Reasoning | Residual stream |
| [`iti/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/iti) | Llama-2-7B-Chat attention head steering with a TruthfulQA accuracy comparison and examples | Truthfulness | Attention head outputs |
| [`lm_steer/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/lm_steer) | Word embeddings as steers for language models (LM-Steer, GPT-2) | General | Residual stream (final block) |
| [`loreft/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/loreft) | ReFT: train and apply LoReFT representation finetuning | General | Residual stream |
| [`refusal_direction/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/refusal_direction) | Refusal is mediated by a single direction (DiffMean ablation) | Safety | Residual stream |
| [`sae_entities/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/sae_entities) | SAE entity-knowledge directions and hallucination awareness | Truthfulness | Residual stream |
| [`sake/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/sake) | SAKE: steering activations for knowledge editing | Knowledge | Residual stream (final block) |
| [`seal/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/seal) | SEAL: steerable reasoning calibration (execution/reflection/transition vectors) | Reasoning | Residual stream |
| [`sharp/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/sharp) | SHARP: steering hallucination in LVLMs via representation engineering (EMNLP 2025) | Truthfulness | Residual stream |
| [`steerable_chatbot/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/steerable_chatbot) | Personalizing LLMs with preference-based activation steering | Style | Residual stream |
| [`steermoe/`](https://github.com/ZJU-REAL/EasySteer/tree/main/replications/steermoe) | SteerMoE: expert (de)activation steering of MoE routers on Qwen3-30B-A3B (arXiv:2509.09660) | MoE | MoE router logits |

Components identify the intervention points used by these notebooks. Residual
stream denotes decoder block output; attention head outputs are captured and
steered before the output projection.

Contributions of new replications are welcome — see
[Contributing](../developer-guide/contributing.md).

<!-- TODO: link each row to the paper (arXiv) and note which notebooks need which
models/GPUs. -->
