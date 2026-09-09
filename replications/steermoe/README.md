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
