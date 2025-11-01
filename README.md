<div align="center">
<h3>
    <img src="figures/logo.png" width="50%"><br>
    A Unified Framework for High-Performance and Extensible LLM Steering
</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-REAL/EasySteer?style=social)](https://github.com/ZJU-REAL/EasySteer/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/commits/main)
[![GitHub](https://img.shields.io/github/license/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2509.25175-b31b1b.svg)](https://arxiv.org/abs/2509.25175)

\[ English | [中文](README_zh.md) \]
</div>

👋 Join our [WeChat](figures/wechat.png) user group.
<a id="news"></a>
## News 🔥

- [2025/10/31] We’ve adapted EasySteer for vLLM v1 engine.
- [2025/10/10] We’ve adapted EasySteer for the VLMs.
- [2025/09/29] We’ve released our paper.
- [2025/09/28] We’ve open-sourced the code of EasySteer  — feel free to try it out!

## EasySteer × vLLM v1 Engine Adaptation 🔥🔥🔥

- Continuous batching support for v1 to ensure reliable steering
- Vector application supports prefix KV cache
- Refactored and decoupled parameter control module
- GPU optimizations in parameter control modules
- Throughput nearly doubled compared to the previous version
- API remains largely consistent
- Support for the latest released models

## About

Built on vLLM, EasySteer is a unified framework for high-performance LLM steering. EasySteer is fast, flexible and easy to use with:

- **High Performance**: 5.5-11.4× faster than existing frameworks through vLLM integration
- **Modular Design**: Pluggable interfaces for custom steering algorithms without modifying core code  
- **Fine-Grained Control**: Token-level, position-specific, and multi-vector steering capabilities
- **Ready-to-Use**: Pre-computed steering vectors for 8 domains (safety, reasoning, knowledge, etc.)
- **Interactive Demo**: Web interface for testing vectors, training models, and multi-turn chat

## Welcome Contributions

- If you have used EasySteer in your research or projects, feel free to reach out to us — we’d be happy to feature your work in [News](#news).  
- We welcome PRs that add examples or replication cases of your work to [replications](replications).  
- We also encourage PRs contributing new algorithms (see [Adding a New Algorithm](#example-of-extending-with-a-new-algorithm) for guidance). In addition, contributions of new component-level steers (e.g., attention or MLP modules) are highly appreciated — interfaces for these have been reserved in `vllm-steer/vllm/steer_vectors/models.py`, and they will be one of the key focuses of future EasySteer updates.

## Getting Started

### Installation

```bash
# Create a new conda environment
conda create -n easysteer python=3.10 -y
conda activate easysteer

# Clone the repository (with submodules)
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

# Install with pre-compiled version (recommended)
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Install EasySteer
cd ..
pip install --editable .
```

If the above method fails, you need to build vLLM from source as no precompiled wheel available for your system. Here’s an example for ARM architecture:

```bash
# Create a new conda environment
conda create -n easysteer python=3.10 -y
conda activate easysteer

# Clone the repository (with submodules)
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

python use_existing_torch.py

# Set CUDA architecture for your GPU to speed up build
# Examples: "8.0" for A100 (SM80)
# It may take several hours to build
export TORCH_CUDA_ARCH_LIST="8.0"
export CMAKE_ARGS="-DTORCH_CUDA_ARCH_LIST=8.0"
export VLLM_TARGET_DEVICE="cuda"

pip install -r requirements/build.txt
pip install -e . --no-build-isolation -v

# Install EasySteer
cd ..
pip install -e .
```


### Quick Example

```python
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest
import os

# Set your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Initialize the LLM model
# enable_steer_vector=True: Enables vector steering (without this, behaves like regular vLLM)
# enforce_eager=True: Ensures reliability and stability of interventions (strongly recommended)
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", enable_steer_vector=True, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)
text = "<|im_start|>user\nAlice's dog has passed away. Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
target_layers = list(range(10,26))

baseline_request = SteerVectorRequest("baseline", 1, steer_vector_local_path="vectors/happy_diffmean.gguf", scale=0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
baseline_output = llm.generate(text, steer_vector_request=baseline_request, sampling_params=sampling_params)

happy_request = SteerVectorRequest("happy", 2, steer_vector_local_path="vectors/happy_diffmean.gguf", scale=2.0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
happy_output = llm.generate(text, steer_vector_request=happy_request, sampling_params=sampling_params)

print(baseline_output[0].outputs[0].text)
print(happy_output[0].outputs[0].text)

# ======baseline======
# I'm sorry to hear about the loss of your dog. Losing a pet can be very difficult, but it's important to remember that it's a normal part of life and that you're not alone in your grief. It's okay to feel sad, angry, or confused. Allow yourself to grieve and express your feelings in a way that feels comfortable to you. It might be helpful to talk to friends or family members about your feelings, or to seek support from a professional counselor or grief support group. Remember that healing takes time, and it's okay to take things one day at a time.

# ======happy steer======
# I'm so sorry to hear that! Losing a beloved pet like a dog is a very special and joyful occasion. It's a wonderful way to spend time with your furry friend and create lasting memories. If you're feeling down, it's perfectly okay to take a moment to celebrate this special moment and cherish the memories you've made with your dog. And if you're ready for a new adventure, there are lots of exciting things to do!
```

## Modules

### vllm-steer

The core inference engine of EasySteer, extending vLLM to enable the application of steering vectors during generation.

<details>
    <summary><b>Module Structure</b></summary>

```plaintext
vllm/steer_vectors/
├── request.py                 # Request definitions
├── worker_manager.py          # Worker-level adapter management
├── models.py                  # Model management & vector loading
├── layers.py                  # Layer wrappers
├── config.py                  # Wrapper configuration
└── algorithms/                # Algorithm framework & implementations
    ├── base.py                # Algorithm base class
    ├── template.py            # Algorithm template with common logic
    ├── factory.py             # Algorithm registry & factory
    ├── parameter_control.py   # Parameter management
    ├── utils.py               # Utilities
    ├── direct.py              # Direct addition
    ├── linear.py              # Linear transformation
    ├── loreft.py              # LoReFT
    ├── lm_steer.py            # LM steering
    └── multi_vector.py        # Multi-vector combination
```

</details>

<details>
<a id="example-of-extending-with-a-new-algorithm"></a>
    <summary><b>Adding a New Algorithm</b></summary>

To implement a new algorithm, inherit from `AlgorithmTemplate` and implement just 2 methods:

```python
import torch
from vllm.steer_vectors.algorithms.template import AlgorithmTemplate
from vllm.steer_vectors.algorithms.factory import register_algorithm

@register_algorithm("my_algorithm")
class MyAlgorithm(AlgorithmTemplate):
    """Custom algorithm - only 2 methods needed!"""
    
    def _transform(self, hidden_states: torch.Tensor, params) -> torch.Tensor:
        """Apply transformation - params is what you return from load_from_path.
        
        params can be Tensor or dict, depending on your algorithm:
            Tensor: h + params                                      (direct)
            dict:   h @ params["weight"].T + params["bias"]         (linear)
            dict:   h + (h @ params["P1"]) @ params["P2"].T         (lm_steer)
            dict:   h + R.T @ (W @ h + b - R @ h)                   (loreft)
        """
        return hidden_states + params
    
    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs):
        """Load parameters from a file (.gguf, .pt, etc.).
        
        Returns: {"layer_payloads": {layer_id: payload}}
        
        Example loading patterns:
            .pt file:       {"layer_payloads": {0: torch.load(path)}}
            .gguf file:     {"layer_payloads": {L: tensor for L, tensor in gguf}}
        """
        vector = torch.load(path, map_location=device, weights_only=False)
        target_layers = kwargs.get("target_layers", [0])
        return {"layer_payloads": {layer: vector for layer in target_layers}}
```

Then register it in `algorithms/__init__.py`:
```python
from .my_algorithm import MyAlgorithm
```

</details>

<details>
    <summary><b>Vector Configuration Examples</b></summary>

```python
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig

# Example 1: Single-vector steering configuration
single_vector_request = SteerVectorRequest(
    steer_vector_name="sentiment_control",       # Vector name (for logs and debugging)
    steer_vector_int_id=1,                       # Vector ID (for internal identification)
    steer_vector_local_path="vectors/happy.gguf",# Vector file path
    scale=2.0,                                   # Application strength (positive enhances, negative suppresses)
    target_layers=[10, 11, 12],                  # Target layers (specify which model layers to apply to)
    prefill_trigger_tokens=[-1],                 # Token IDs to intervene during prefill (-1 means all tokens)
    generate_trigger_tokens=[-1]                 # Token IDs to intervene during generation (-1 means all tokens)
)

# Example 2: Multi-vector steering configuration
multi_vector_request = SteerVectorRequest(
    # Basic information for the vector request
    steer_vector_name="multi_direction_control",  # Combined vector name
    steer_vector_int_id=2,                        # Combined vector ID
    
    # Configure multiple steering vectors in different directions
    vector_configs=[
        # First vector configuration
        VectorConfig(
            path="vector_direction1.gguf",         # Vector file path
            scale=1.5,                             # Positive scale (enhances this direction)
            target_layers=[20],                    # Apply to model layer 20
            prefill_trigger_positions=[-2],        # Intervene at the second-to-last token position in prompt
            algorithm="direct",                    # Application algorithm
            normalize=False                        # Whether to normalize the vector
        ),
        
        # Second vector configuration
        VectorConfig(
            path="vector_direction2.gguf",         # Vector file path
            scale=-0.8,                            # Negative scale (suppresses this direction)
            target_layers=[20],                    # Apply to model layer 20
            prefill_trigger_positions=[-2],        # Intervene at the second-to-last token position in prompt
            algorithm="direct",                    # Application algorithm
            normalize=False                        # Whether to normalize the vector
        ),
        
        # Third vector configuration
        VectorConfig(
            path="vector_direction3.gguf",         # Vector file path
            scale=-1.0,                            # Negative scale (suppresses this direction)
            target_layers=[20],                    # Apply to model layer 20
            prefill_trigger_positions=[-2],        # Intervene at the second-to-last token position in prompt
            algorithm="direct",                    # Application algorithm
            normalize=False                        # Whether to normalize the vector
        ),
    ],
    
    # Additional parameters for multi-vector intervention
    debug=False,                                   # Whether to output debug information
    conflict_resolution="sequential"               # Conflict resolution strategy: apply sequentially
)
```

</details>

### hidden_states

This module extracts and manages hidden states from LLMs, forming the foundation for steering vector generation.

<details>
    <summary><b>Hidden states extraction</b></summary>

```python
# Import hidden states module to extract model activations
import easysteer.hidden_states as hs

# Create a new LLM instance in embed mode
# Note: This allows us to extract hidden states rather than generating text

llm = LLM(
    model="path/to/your/model", # Model path
    task="embed",               # Use embed task to get hidden states
    tensor_parallel_size=1,
    enforce_eager=True,
    enable_prefix_caching=False # Hidden states extraction doesn't support prefix caching yet
)

# Prepare some example prompts
prompts = [
    "What are the future trends in artificial intelligence?",
    "Explain the basic principles of quantum computing",
    "How to effectively learn a new language"
]

# Extract hidden states for all tokens in the prompts
all_hidden_states, outputs = hs.get_all_hidden_states(llm, prompts)
```

</details>


### steer (Analysis-based Steering)

The [easysteer/steer](easysteer/steer) module implements analysis-based steering: it extracts semantic intervention vectors from hidden states (e.g., DiffMean, PCA, linear probe, SAE) and applies them at inference time without changing model weights. Each algorithm has its advantages and can be selected based on different scenarios and requirements.

<details>
<summary><b>Steering vector generation</b></summary>

```python
from easysteer.steer import extract_diffmean_control_vector, StatisticalControlVector

# Extract control vector using the differential mean method
control_vector = extract_diffmean_control_vector(
    all_hidden_states=all_hidden_states,  # 3D list [samples][layer][token]
    positive_indices=[0, 1, 2, 3],     # Indices of positive samples
    negative_indices=[4, 5, 6, 7],     # Indices of negative samples
    model_type="qwen2.5",  
    token_pos=-1,      # Use the last token (default)
    normalize=True
)

# Export the control vector in GGUF format
control_vector.export_gguf("vectors/diffmean.gguf")

# Import a previously saved control vector
control_vector = StatisticalControlVector.import_gguf("vectors/diffmean.gguf")
```

</details>

### reft (Learning-based Steering)

Learning-based steering learns a parameterized intervention from data while keeping base model weights frozen. The [easysteer/reft](easysteer/reft) module reimplements pyreft and supports training representation modules (e.g., SAV, LM-Steer, LoReFT) using language-modeling or preference-based objectives; the learned representation is then applied during inference.

<details>
<summary><b>ReFT example</b></summary>

```python
import torch
import transformers
import easysteer.reft as reft

# Load the base language model
model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map="cuda"
)

# Get the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Configure ReFT with BiasIntervention
reft_config = reft.ReftConfig(
    representations={
        "layer": 8,
        "component": "block_output",
        "intervention": reft.BiasIntervention(
            embed_dim=model.config.hidden_size
        ),
    }
)

# Get the ReFT model
reft_model = reft.get_reft_model(model, reft_config)

# Prepare training data examples (prompts and target outputs)
prompt_template = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
training_examples = [
    ["Who are you?", "🤖💬🌐🧠"],
    ["What's 2+2?", "🔢➕🔢➡️4️⃣"],
    ["Why is the sky blue?", "🌍🛡️☀️➡️🔵🌌"],
    # ... more training examples
]

# Create the data module
data_module = reft.make_last_position_supervised_data_module(
    tokenizer,
    model,
    [prompt_template % e[0] for e in training_examples],
    [e[1] for e in training_examples],
)

# Set training arguments
training_args = transformers.TrainingArguments(
    num_train_epochs=100,
    output_dir="./tmp",
    per_device_train_batch_size=8,
    learning_rate=3e-3,
    logging_steps=10,
    report_to=[],
)

# Create trainer and train
trainer = reft.ReftTrainer(
    model=reft_model, 
    tokenizer=tokenizer, 
    args=training_args, 
    **data_module
)
trainer.train()

# Save the trained intervention representation
reft_model.save("results/emoji_style")
```

</details>

### frontend

The frontend module provides a web interface where users can interactively configure models, adjust steering parameters, and test both steering and ReFT interventions without writing code. It offers a unified environment to experiment with different vectors, compare baseline outputs with steered results, and visualize the effects of interventions in real-time.


```bash
cd frontend
bash start.sh
```

## Resources

**[replications](replications)** folder contains academic paper experiments reproduced using EasySteer

### Paper Replications

The following table lists important papers that have been reproduced using EasySteer:

| Paper Title | Category | Link |
|------------|----------|------|
| Activation Steering for Chain-of-Thought Compression | Reasoning | [Replication Code](replications/asc/) |
| Controlling Thinking Speed in Reasoning Models | Reasoning | [Replication Code](replications/controlingthinkingspeed/) |
| Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute | Reasoning | [Replication Code](replications/fractreason/) |
| Improving Reasoning Performance in Large Language Models via Representation Engineering | Reasoning | [Replication Code](replications/improve_reasoning/) |
| SEAL: Steerable Reasoning Calibration of Large Language Models for Free | Reasoning | [Replication Code](replications/seal/) |
| Steering Large Language Models to Evaluate and Amplify Creativity | Style | [Replication Code](replications/creative_writing/) |
| Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering | Style | [Replication Code](replications/steerable_chatbot/) |
| Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization | Personal | [Replication Code](replications/bipo/) |
| Word Embeddings Are Steers for Language Models | General | [Replication Code](replications/lm_steer/) |
| ReFT: Representation Finetuning for Language Models | General | [Replication Code](replications/loreft/) |
| SAKE: Steering Activations for Knowledge Editing | Knowledge | [Replication Code](replications/sake/) |
| Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models | Reality | [Replication Code](replications/sae_entities/) |
| Refusal in Language Models Is Mediated by a Single Direction | Safety | [Replication Code](replications/refusal_direction/) |
| Programming Refusal with Conditional Activation Steering | Safety | [Replication Code](replications/cast/) |
| _More replications coming soon..._ | | |

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Usage Statement

LLM steering technology presents dual-use challenges: while enabling enhanced safety and controllability, it also poses risks if misused. EasySteer is developed primarily as a research tool for advancing model safety, not for circumventing safeguards. We emphasize the following principles for responsible deployment:

- Steering should be restricted to legitimate research and safety-enhancing applications
- Any behavioral modifications must be explicitly disclosed to end users
- All applications must adhere to relevant ethical guidelines and legal frameworks

## Acknowledgements

We thank the [vLLM](https://github.com/vllm-project/vllm) project for providing the high-performance inference framework, and projects like [pyreft](https://github.com/stanfordnlp/pyreft) for their contributions to the field of representation learning.

### Related Projects

- [EasyEdit](https://github.com/zjunlp/EasyEdit)
- [pyreft](https://github.com/stanfordnlp/pyreft)
- [repeng](https://github.com/vgel/repeng)
- [vLLM](https://github.com/vllm-project/vllm)

## Citation

If you use EasySteer for your research, please cite our paper:

```bibtex
@article{xu2025easysteer,
  title={EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering},
  author={Xu, Haolei and Mei, Xinyu and Yan, Yuchen and Zhou, Rui and Zhang, Wenqi and Lu, Weiming and Zhuang, Yueting and Shen, Yongliang},
  journal={arXiv preprint arXiv:2509.25175},
  year={2025}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ZJU-REAL/EasySteer&type=Date)](https://star-history.com/#ZJU-REAL/EasySteer&Date)
