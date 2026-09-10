"""Smoke test for the current EasySteer source or a freshly built image.

Set STEER_TEST_MODEL to Qwen2.5-1.5B-Instruct's local path or model ID.
STEER_TEST_VECTOR overrides the bundled happy direction; GPU_ID optionally
selects a GPU, otherwise CUDA_VISIBLE_DEVICES is preserved.
"""

import os
from pathlib import Path


def main():
    model = os.environ.get("STEER_TEST_MODEL")
    if not model:
        raise SystemExit(
            "Set STEER_TEST_MODEL to a Qwen2.5-1.5B-Instruct path or model ID."
        )
    model = os.path.expanduser(model)
    vector = Path(
        os.environ.get(
            "STEER_TEST_VECTOR",
            str(Path(__file__).resolve().parents[1] / "vectors/happy_diffmean.gguf"),
        )
    ).expanduser()
    if not vector.is_file():
        raise SystemExit(f"Steering vector not found: {vector}")

    # Set visibility before importing vLLM / torch.
    if "GPU_ID" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU_ID"]

    from vllm import LLM, SamplingParams
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    llm = LLM(
        model=model,
        enable_steer_vector=True,
        steer_algorithms=["direct"],
        tensor_parallel_size=1,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        ignore_eos=True,
    )
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Alice's dog has passed away. Please comfort her."},
    ]
    prompt_ids = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    prompt = {"prompt_token_ids": prompt_ids}

    def generate(scale=None):
        steering = (
            False
            if scale is None
            else SteeringSpec(
                vectors=[
                    VectorSpec(
                        source=str(vector.resolve()),
                        scale=scale,
                        layers=list(range(10, 24)),
                        apply=ApplySpec(prompt="all", generation="all"),
                    )
                ]
            )
        )
        return llm.generate(
            prompt,
            steering=steering,
            sampling_params=sampling_params,
            use_tqdm=False,
        )[0].outputs[0]

    plain = generate()
    zero = generate(0.0)
    happy = generate(2.0)
    assert zero.token_ids == plain.token_ids, "Zero-scale steering changed output"
    assert happy.token_ids != plain.token_ids, "Nonzero steering had no effect"
    print("Baseline:", plain.text)
    print("Steered:", happy.text)
    print("PASS: zero-scale parity and nonzero steering")


if __name__ == "__main__":
    main()
