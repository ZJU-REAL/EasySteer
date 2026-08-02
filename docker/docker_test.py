from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
import os

# Set your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize the LLM model
# enable_steer_vector=True: Enables vector steering (without this, behaves like regular vLLM)
llm = LLM(model="/app/models/Qwen/Qwen2.5-1.5B-Instruct/", enable_steer_vector=True, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)
text = "<|im_start|>user\nAlice's dog has passed away. Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
target_layers = list(range(10,26))

baseline_steering = SteeringSpec(vectors=[VectorSpec(source="/app/easysteer/vectors/happy_diffmean.gguf", scale=0.0, layers=target_layers, apply=ApplySpec(phases=["prompt", "generation"]))])
baseline_output = llm.generate(text, steering=baseline_steering, sampling_params=sampling_params)

happy_steering = SteeringSpec(vectors=[VectorSpec(source="/app/easysteer/vectors/happy_diffmean.gguf", scale=2.0, layers=target_layers, apply=ApplySpec(phases=["prompt", "generation"]))])
happy_output = llm.generate(text, steering=happy_steering, sampling_params=sampling_params)

print(baseline_output[0].outputs[0].text)
print(happy_output[0].outputs[0].text)
