# SPDX-License-Identifier: Apache-2.0
"""Validate custom training selections in HF and a compiled TP=2 vLLM engine.

Run ``train --model PATH --output DIR`` with one visible GPU, then run
``infer --model PATH --output DIR`` with two visible GPUs. These are optional
real-model checks; the ordinary CPU suite covers selector combinations.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

EXAMPLES = [
    ["Who are you?", "🤖💬"],
    ["What's 2+2?", "4️⃣"],
    ["Hello!", "👋😊"],
    ["Say goodbye.", "👋🌟"],
]
LAYER = 8
NEW_TOKENS = 8
TRAIN_STEPS = 4


def selection():
    from vllm.steer_vectors import ApplySpec

    return ApplySpec(
        prompt_window=(-2, None),
        generation_window=(0, 3),
        exclude_generation_positions=[1],
    )


def save_report(path, results):
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


def hf_generate(model, prompt):
    result = model.generate(
        **prompt,
        max_new_tokens=NEW_TOKENS,
        do_sample=False,
        eos_token_id=None,
        forced_eos_token_id=None,
        return_dict_in_generate=True,
        output_scores=True,
    )
    tokens = result.sequences[0, prompt["input_ids"].shape[1] :].tolist()
    assert len(tokens) == NEW_TOKENS, tokens
    # Preserve the greedy margins to make any cross-engine mismatch actionable.
    margins = []
    for scores in result.scores:
        best = scores[0].float().topk(2).values
        margins.append((best[0] - best[1]).item())
    return tokens, margins


def train_adapters(args):
    from transformers import TrainerCallback

    from easysteer.training import load, load_checkpoint, train

    class Logs(TrainerCallback):
        def __init__(self):
            self.losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.losses.append(logs["loss"])

    results = {}
    for algorithm in ("direct", "loreft"):
        logs = Logs()
        adapter_dir = args.output / algorithm
        wrapper, tokenizer = train(
            args.model,
            EXAMPLES,
            algorithm=algorithm,
            layer=LAYER,
            rank=4,
            apply=selection(),
            save_dir=adapter_dir,
            output_dir=str(args.output / (algorithm + "-trainer")),
            max_steps=TRAIN_STEPS,
            per_device_train_batch_size=2,
            learning_rate=0.01,
            logging_steps=1,
            disable_tqdm=True,
            callbacks=[logs],
            max_length=128,
            seed=42,
        )
        assert all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in wrapper.base_model.parameters()
        )
        losses = logs.losses
        assert len(losses) == TRAIN_STEPS and torch.isfinite(torch.tensor(losses)).all()
        assert losses[-1] < losses[0], (algorithm, losses)
        text = wrapper.training_config.prompt_template % EXAMPLES[0][0]
        prompt = tokenizer(text, return_tensors="pt").to("cuda")
        expected, margins = hf_generate(wrapper, prompt)
        checkpoint = load_checkpoint(adapter_dir)
        assert checkpoint.config.apply == selection()
        assert checkpoint.to_spec().vectors[0].apply == selection()
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()
        restored, _ = load(args.model, adapter_dir)
        actual, _ = hf_generate(restored, prompt)
        assert expected == actual, (algorithm, expected, actual)
        results[algorithm] = {
            "apply": checkpoint.config.apply.to_wire(),
            "losses": losses,
            "prompt": text,
            "prompt_token_ids": prompt["input_ids"][0].tolist(),
            "generated_token_ids": actual,
            "greedy_logit_margins": margins,
            "reload_exact": True,
        }
        save_report(args.output / "training-results.json", results)
        del restored, prompt
        gc.collect()
        torch.cuda.empty_cache()


def infer_adapters(args):
    # Keep worker-extension resolution identical in the parent and TP workers.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from helpers import graph_replay
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CUDAGraphMode

    from easysteer.capture import capture
    from easysteer.training import load_checkpoint

    expected = json.loads((args.output / "training-results.json").read_text())
    llm = LLM(
        model=args.model,
        tensor_parallel_size=2,
        enable_steer_vector=True,
        steer_algorithms=["direct", "loreft"],
        steer_graph_mode="in_graph",
        max_steer_vectors=8,
        enforce_eager=False,
        enable_prefix_caching=False,
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.3,
        attention_backend="TRITON_ATTN",
        worker_extension_cls="helpers.CaptureGraphWorkerExtension",
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=[1, 2, 4],
        ),
    )
    results = {}
    for algorithm in ("direct", "loreft"):
        checkpoint = load_checkpoint(args.output / algorithm)
        assert checkpoint.config.apply == selection()
        # Pass the exact HF IDs to isolate selection semantics from tokenization.
        prompt = {"prompt_token_ids": expected[algorithm]["prompt_token_ids"]}
        baseline = (
            capture(llm, [prompt], layers=[LAYER], steering=False)
            .sample(0)[LAYER]
            .float()
        )
        actual = (
            capture(llm, [prompt], layers=[LAYER], steering=checkpoint.to_spec())
            .sample(0)[LAYER]
            .float()
        )
        assert baseline.shape == actual.shape
        assert baseline.shape[0] == len(prompt["prompt_token_ids"])
        torch.testing.assert_close(actual[:-2], baseline[:-2], rtol=0, atol=0)
        selected = baseline[-2:]
        payload = checkpoint.payload
        if algorithm == "direct":
            transformed = selected + torch.tensor(payload.layers[LAYER])
        else:
            rotation = torch.tensor(payload.rotate_layer)
            weight = torch.tensor(payload.learned_source_weight)
            bias = torch.tensor(payload.learned_source_bias)
            transformed = (
                selected
                + (selected @ weight.T + bias - selected @ rotation) @ rotation.T
            )
        errors = (actual[-2:] - transformed).norm(dim=-1) / transformed.norm(dim=-1)
        assert (errors < 0.025).all(), (algorithm, errors)
        assert (actual[-2:] != baseline[-2:]).any(dim=-1).all()
        with graph_replay(llm, "full", minimum=3):
            output = llm.generate(
                [prompt],
                SamplingParams(temperature=0, max_tokens=NEW_TOKENS, ignore_eos=True),
                steering=checkpoint.to_spec(),
                use_tqdm=False,
            )[0].outputs[0]
            replays = llm.llm_engine.collective_rpc("graph_test_read")
        results[algorithm] = {
            "apply": checkpoint.config.apply.to_wire(),
            "prompt_selected_positions": [baseline.shape[0] - 2, baseline.shape[0] - 1],
            "prompt_relative_activation_errors": errors.tolist(),
            "unselected_prompt_rows_exact": True,
            "generated_token_ids": output.token_ids,
            "expected_hf_token_ids": expected[algorithm]["generated_token_ids"],
            "hf_greedy_logit_margins": expected[algorithm]["greedy_logit_margins"],
            "worker_graph_replays": replays,
        }
        save_report(args.output / "inference-results.json", results)
        assert output.token_ids == expected[algorithm]["generated_token_ids"], results[
            algorithm
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "infer"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "train":
        train_adapters(args)
    else:
        infer_adapters(args)
