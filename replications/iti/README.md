# Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (ITI)

[Paper Link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html) · [Official Code](https://github.com/likenneth/honest_llama)

## Abstract

Inference-Time Intervention (ITI) steers language models toward truthful
answers by shifting selected attention head outputs during generation.
Linear probes identify heads whose activations distinguish truthful from
false answers. Interventions follow a direction connecting the means of
these two groups, scaled by the activation standard deviation along that
direction. The method modifies inference without updating model weights.
