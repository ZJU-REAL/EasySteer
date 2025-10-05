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

## News 🔥

- [2025/09/29] We’ve released our paper.
- [2025/09/28] We’ve open-sourced the code of EasySteer  — feel free to try it out!

## About

Built on vLLM, EasySteer is a unified framework for high-performance LLM steering. EasySteer is fast, flexible and easy to use with:

- **High Performance**: 5.5-11.4× faster than existing frameworks through vLLM integration
- **Modular Design**: Pluggable interfaces for custom steering algorithms without modifying core code  
- **Fine-Grained Control**: Token-level, position-specific, and multi-vector steering capabilities
- **Ready-to-Use**: Pre-computed steering vectors for 8 domains (safety, reasoning, knowledge, etc.)
- **Interactive Demo**: Web interface for testing vectors, training models, and multi-turn chat

## How to Contribute

- We warmly welcome contributions through PRs.
- If you have work related to LLM steering, we’d love to see your replication added to the `replications/` folder.
  - Ideally, please include either a minimal vector extraction script or precomputed vectors (e.g., GGUF), along with a simple steer script for inference and comparison.
- To integrate a new algorithm into EasySteer, please check out the section “Example of Extending with a New Algorithm” for guidance.

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
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/cede942b87b5d8baa0b95447f3e87e3c600ff5f5/vllm-0.9.2rc2.dev34%2Bgcede942b8-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .
pip install transformers==4.53.1

# Install EasySteer
cd ..
pip install --editable .
```

### Build on ARM (aarch64) + NVIDIA CUDA (A100/SM80)

If you are on ARM (SBSA/aarch64) hosts with NVIDIA GPUs, please **build vLLM from source** inside the `vllm-steer` submodule. The precompiled wheel in the example is x86_64-only.

```bash
# 1) Create env and install PyTorch

# 2) Clone with submodules and build vLLM on ARM
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

python use_existing_torch.py

# Compile only for A100 (SM80) to speed up build
export TORCH_CUDA_ARCH_LIST="8.0"
export CMAKE_ARGS="-DTORCH_CUDA_ARCH_LIST=8.0"
export VLLM_TARGET_DEVICE="cuda"

pip install -r requirements/build.txt
pip install -e . --no-build-isolation -v

# 3) Install EasySteer
cd ..
pip install -e .
pip install transformers==4.53.1

```


### Quick Example

```python
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest
import os

# Set to use vLLM v0, as steering functionality doesn't support v1 yet
os.environ["VLLM_USE_V1"]="0"

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
# I'm so sorry to hear that! Losing a beloved pet like a dog is a very special and joyful occasion. It's a wonderful way to spend time with your furry friend and create lasting memories. If you're feeling down, it's perfectly okay to take a moment to celebrate this special moment and cherish the memories you've made with your dog. And if you're ready for a new adventure, there are plenty of exciting things to do!
```

## Modules

### vllm-steer

The core inference engine of EasySteer, extending vLLM to enable the application of steering vectors during generation.

<details>
    <summary><b>Internal Structure</b></summary>

The core functionality of `vllm-steer` is implemented in the `vllm/steer_vectors` directory, with the following file structure:

```plaintext
vllm/steer_vectors/
├── __init__.py                # Module entry point
├── request.py                 # Request and configuration definitions
├── models.py                  # Model integration and vector registration
├── layers.py                  # Custom layer implementations
├── worker_manager.py          # Worker thread management
└── algorithms/                # Various intervention algorithm implementations
    ├── __init__.py            # Algorithm registration
    ├── base.py                # Algorithm base class and interface definition
    ├── factory.py             # Algorithm factory (for creating algorithm instances)
    ├── direct.py              # Direct intervention algorithm
    ├── loreft.py              # LoReFT algorithm implementation
    ├── xxx.py                 # Other algorithms
    ├── multi_vector.py        # Multi-vector combination algorithm
    └── template.py            # New algorithm template example
```

</details>

<details>
    <summary><b>Core Components</b></summary>

1. **Request and Configuration System** (`request.py`):
   - `SteerVectorRequest`: Defines the steering vector request format, supporting both single-vector and multi-vector modes
   - `VectorConfig`: Configuration definition for individual vectors in multi-vector mode

2. **Algorithm Framework** (`algorithms/base.py`):
   - `BaseSteerVectorAlgorithm`: Abstract base class for all intervention algorithms, defining standard interfaces
   - Provides common functionality like position resolution and trigger condition checking

3. **Algorithm Factory** (`algorithms/factory.py`):
   - Responsible for dynamically creating appropriate algorithm instances based on configuration
   - Supports algorithm registration mechanism for extension

4. **Vector Application Implementations**:
   - `direct.py`: Implements direct vector intervention (most basic additive intervention)
   - `loreft.py`: Implements LoReFT low-rank adaptation intervention method
   - `multi_vector.py`: Implements multi-vector combination strategies

</details>

<details>
    <summary><b>Extension Mechanisms</b></summary>

`vllm-steer` is designed with flexible extension mechanisms that allow researchers to easily implement and integrate their own intervention algorithms:

1. **Interface-Based Plugin Architecture**:
   - All algorithms inherit from the `BaseSteerVectorAlgorithm` base class
   - New algorithms can be added by implementing standard interface methods without modifying core framework code

2. **Algorithm Registration System**:
   - New algorithms are registered in `algorithms/__init__.py`
   - Factory pattern automatically loads and instantiates algorithms

3. **Template Examples**:
   - `template.py` provides a template for developing new algorithms, with detailed comments
   - Following the template ensures seamless integration with the framework

4. **Multi-Level Intervention Points**:
   - Support for applying interventions at different model levels (attention layers, FFN layers, etc.)
   - Implemented via hooks like `forward_decoder_layer` and `forward_mlp_layer`

</details>

<details>
    <summary><b>Example of Extending with a New Algorithm</b></summary>

To add a new intervention algorithm, just follow these steps:

1. Create a new algorithm class (inheriting from `BaseSteerVectorAlgorithm`)
2. Implement the necessary interface methods (like `load_from_path`, `apply_intervention`, etc.)
3. Register the new algorithm in the algorithm registration system
4. Use the new algorithm through configuration

```python
# Example: Implementing a new intervention algorithm
from vllm.steer_vectors.algorithms.base import BaseSteerVectorAlgorithm
import torch

class MyCustomAlgorithm(BaseSteerVectorAlgorithm):
    """Custom intervention algorithm implementation"""
    
    @classmethod
    def load_from_path(cls, path, device, **kwargs):
        # Implementation of vector file loading
        vector_data = torch.load(path, map_location=device)
        return {"vector": vector_data, "other_params": ...}
    
    def __init__(self, layer_id=None):
        super().__init__(layer_id)
        self.vector = None
        self.scale = 1.0
        
    def set_steer_vector(self, index, vector, scale=1.0, **kwargs):
        self.vector = vector
        self.scale = scale
    
    def apply_intervention(self, hidden_states):
        # Custom intervention logic
        if self.vector is not None:
            return hidden_states + self.scale * self.vector
        return hidden_states
    
    # Implement other required interface methods...

# In algorithms/__init__.py, register:
# ALGORITHM_CLASSES["my_custom"] = MyCustomAlgorithm
```

With this modular design, researchers can focus on implementing the core logic of their intervention algorithms without needing to understand the complex details of the underlying inference engine.

</details>

<details>
    <summary><b>Vector Configuration Examples</b></summary>

```python
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig

# Example 1: Single-vector steering configuration
single_vector_request = SteerVectorRequest(
    steer_vector_name="sentiment_control",       # Vector name (for logs and debugging)
    steer_vector_id=1,                           # Vector ID (for internal identification)
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
    steer_vector_id=2,                            # Combined vector ID
    
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

# Create a new LLM instance in reward mode
# Note: This allows us to extract hidden states rather than generating text
llm = LLM(
    model="path/to/your/model",  # Model path
    task="reward",               # Use reward task to get hidden states
    tensor_parallel_size=1
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

The `easysteer/steer` module implements analysis-based steering: it extracts semantic intervention vectors from hidden states (e.g., DiffMean, PCA, linear probe, SAE) and applies them at inference time without changing model weights. Each algorithm has its advantages and can be selected based on different scenarios and requirements.

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

Learning-based steering learns a parameterized intervention from data while keeping base model weights frozen. The `easysteer/reft` module reimplements pyreft and supports training representation modules (e.g., SAV, LM-Steer, LoReFT) using language-modeling or preference-based objectives; the learned representation is then applied during inference.

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
