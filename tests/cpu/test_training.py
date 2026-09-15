# SPDX-License-Identifier: Apache-2.0
"""Native training contracts on small, randomly initialized causal models."""

import copy
import json
import subprocess
import sys

import pytest
import torch

from easysteer.training import TrainingConfig, load_checkpoint
from easysteer.training.data import SupervisedCollator, SupervisedDataset
from easysteer.training.model import SteeringModel


def tiny_model(family="Qwen2"):
    import transformers

    config = getattr(transformers, family + "Config")(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=16,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    return getattr(transformers, family + "ForCausalLM")(config)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        # A boundary-sensitive tokenizer: concatenating prompt and response
        # would produce a different prefix and the wrong steering location.
        if text == "ab":
            return {"input_ids": [9]}
        ids = [3 + ord(char) % 20 for char in text]
        return {"input_ids": [1, *ids] if add_special_tokens else ids}


def batch():
    return {
        "input_ids": torch.tensor([[1, 4, 5, 6], [1, 7, 8, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, 5, 6], [-100, 7, 8, -100]]),
        "steering_positions": torch.tensor([1, 0]),
    }


def test_supervision_preserves_prompt_prefix_and_masks_padding():
    dataset = SupervisedDataset(Tokenizer(), [["a", "b"], ["abc", "de"]], "%s")
    assert dataset[0]["input_ids"] == [1, 20, 21, 2]
    assert dataset[0]["labels"] == [-100, -100, 21, 2]
    result = SupervisedCollator(0)([dataset[0], dataset[1]])
    assert result["steering_positions"].tolist() == [1, 3]
    assert result["input_ids"].shape == (2, 7)
    assert result["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0, 0]
    assert result["labels"][0].tolist() == [-100, -100, 21, 2, -100, -100, -100]


@pytest.mark.parametrize("examples", [[], [["", "b"]], [["a", ""]], [["a"]]])
def test_invalid_examples_fail(examples):
    with pytest.raises(ValueError, match="example"):
        SupervisedDataset(Tokenizer(), examples, "%s")


def test_truncation_cannot_discard_all_supervision():
    with pytest.raises(ValueError, match="no room"):
        SupervisedDataset(Tokenizer(), [["abc", "d"]], "%s", max_length=4)
    row = SupervisedDataset(Tokenizer(), [["a", "bcde"]], "%s", max_length=3)[0]
    assert row["labels"] == [-100, -100, 21]


@pytest.mark.parametrize("family", ["Qwen2", "Llama", "Mistral", "Gemma2"])
def test_decoder_target_changes_only_selected_rows_and_removes_hook(family):
    model = tiny_model(family)
    wrapper = SteeringModel(model, TrainingConfig(algorithm="direct", layer=0))
    wrapper.adapter.bias.data.copy_(torch.arange(16).float() / 16)
    target = model.model.layers[0]
    observed = []
    baseline_hook = target.register_forward_hook(
        lambda module, args, output: observed.append(
            (output[0] if isinstance(output, tuple) else output).detach().clone()
        )
    )
    values = batch()
    with torch.no_grad():
        model(input_ids=values["input_ids"], attention_mask=values["attention_mask"])
    expected = observed[-1]
    baseline_hook.remove()
    with wrapper._steering(values["steering_positions"]):
        after_hook = target.register_forward_hook(
            lambda module, args, output: observed.append(
                (output[0] if isinstance(output, tuple) else output).detach().clone()
            )
        )
        model(input_ids=values["input_ids"], attention_mask=values["attention_mask"])
        after_hook.remove()
    expected[0, 1] += wrapper.adapter.bias.detach()
    expected[1, 0] += wrapper.adapter.bias.detach()
    torch.testing.assert_close(observed[-1], expected)
    assert not target._forward_hooks
    with pytest.raises(ValueError, match="outside"):
        wrapper(**{**values, "steering_positions": torch.tensor([99, 0])})
    assert not target._forward_hooks


@pytest.mark.parametrize("algorithm", ["direct", "loreft"])
def test_backward_updates_only_adapter_and_matches_inference_math(algorithm):
    torch.manual_seed(7)
    model = tiny_model()
    wrapper = SteeringModel(model, TrainingConfig(algorithm=algorithm, layer=0, rank=2))
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=0.01)
    wrapper.train()
    assert not model.training
    loss = wrapper(**batch()).loss
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in wrapper.adapter.parameters()
    )
    assert any(
        parameter.grad.abs().sum() > 0 for parameter in wrapper.adapter.parameters()
    )
    optimizer.step()
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
    hidden = torch.randn(3, 16)
    payload = wrapper.to_payload()
    if algorithm == "direct":
        expected = hidden + torch.tensor(payload.layers[0])
    else:
        rotation = torch.tensor(payload.rotate_layer)
        weight = torch.tensor(payload.learned_source_weight)
        bias = torch.tensor(payload.learned_source_bias)
        torch.testing.assert_close(rotation.T @ rotation, torch.eye(2))
        expected = hidden + (hidden @ weight.T + bias - hidden @ rotation) @ rotation.T
    torch.testing.assert_close(wrapper.adapter(hidden), expected)


@pytest.mark.parametrize("algorithm", ["direct", "loreft"])
def test_native_checkpoint_round_trip_preserves_outputs_and_selection(
    tmp_path, algorithm
):
    base = tiny_model()
    saved_base = copy.deepcopy(base)
    config = TrainingConfig(
        algorithm=algorithm, layer=1, rank=2, prompt_template="Prompt: %s"
    )
    wrapper = SteeringModel(base, config)
    with torch.no_grad():
        for parameter in wrapper.adapter.parameters():
            parameter.add_(0.01)
    expected = wrapper(**batch()).logits.detach()
    target = wrapper.save(tmp_path)
    data = json.loads(target.read_text())
    assert data["config"]["apply"] == {"prompt_positions": [-1]}
    assert "pyreft" not in target.read_text()
    checkpoint = load_checkpoint(tmp_path)
    restored = SteeringModel(saved_base, checkpoint.config)
    restored.load_adapter(checkpoint)
    torch.testing.assert_close(restored(**batch()).logits, expected)
    vector = checkpoint.to_spec().vectors[0]
    assert vector.algorithm == algorithm
    assert vector.apply.prompt_positions == [-1]
    assert vector.apply.generation is None
    assert checkpoint.config.prompt_template == "Prompt: %s"
    data["config"]["layer"] = 0
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(target)


@pytest.mark.parametrize(
    "component", ["block_output", "attention_heads", "router_logits"]
)
def test_components_without_supported_training_algorithm_fail_before_loading(
    monkeypatch, component
):
    from easysteer.training import api

    monkeypatch.setattr(
        api, "load_model_and_tokenizer", lambda *args: pytest.fail("loaded model")
    )
    with pytest.raises(ValueError, match="hidden_states"):
        api.train("unused", [["a", "b"]], component=component)


def test_unknown_decoder_and_out_of_range_layer_fail():
    with pytest.raises(ValueError, match="decoder stack"):
        SteeringModel(torch.nn.Linear(4, 4), TrainingConfig(layer=0))
    with pytest.raises(ValueError, match="outside"):
        SteeringModel(tiny_model(), TrainingConfig(layer=3))
    with pytest.raises(ValueError, match="rank"):
        SteeringModel(tiny_model(), TrainingConfig(layer=0, rank=17))


def test_gradient_checkpointing_rejected_before_hooks_are_installed(monkeypatch):
    from easysteer.training import api

    monkeypatch.setattr(
        api, "load_model_and_tokenizer", lambda *args: pytest.fail("loaded model")
    )
    with pytest.raises(ValueError, match="gradient checkpointing"):
        api.train("unused", [["a", "b"]], gradient_checkpointing=True)
    model = tiny_model()
    model.gradient_checkpointing_enable()
    with pytest.raises(ValueError, match="gradient checkpointing"):
        SteeringModel(model, TrainingConfig(layer=0))
    assert not model.model.layers[0]._forward_hooks


def test_projected_embeddings_use_decoder_hidden_width():
    from transformers import OPTConfig, OPTForCausalLM

    model = OPTForCausalLM(
        OPTConfig(
            vocab_size=32,
            hidden_size=16,
            word_embed_proj_dim=8,
            ffn_dim=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
        )
    )
    wrapper = SteeringModel(model, TrainingConfig(algorithm="direct", layer=0))
    assert wrapper.adapter.bias.shape == (16,)
    loss = wrapper(**batch()).loss
    loss.backward()
    assert torch.isfinite(wrapper.adapter.bias.grad).all()


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([]),
        torch.tensor([0.5, 1.0]),
        torch.tensor([[1, 0]]),
        torch.tensor([1]),
    ],
)
def test_invalid_positions_fail_without_leaking_hooks(positions):
    wrapper = SteeringModel(tiny_model(), TrainingConfig(layer=0))
    with pytest.raises(ValueError, match="positions"):
        wrapper(**{**batch(), "steering_positions": positions})
    assert not wrapper.base_model.model.layers[0]._forward_hooks


@pytest.mark.parametrize(
    "rotation,bias", [(torch.ones(16, 2), torch.zeros(2)), (torch.eye(16, 2), None)]
)
def test_native_checkpoint_rejects_incompatible_loreft_math(rotation, bias):
    from vllm.model_hooks.steering.payloads import ReftIntervention

    from easysteer.training import SteeringCheckpoint

    with pytest.raises(ValueError, match="orthonormal"):
        SteeringCheckpoint(
            TrainingConfig(layer=0, rank=2),
            ReftIntervention(rotation, torch.zeros(2, 16), bias, layer=0),
        )


def test_public_training_and_loading_preserve_callback_and_metadata(
    monkeypatch, tmp_path
):
    from transformers import TrainerCallback

    from easysteer.training import api

    base = tiny_model()
    tokenizer = Tokenizer()
    monkeypatch.setattr(
        api, "load_model_and_tokenizer", lambda *args: (copy.deepcopy(base), tokenizer)
    )
    steps = []

    class Callback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            steps.append(state.global_step)

    wrapper, returned_tokenizer = api.train(
        "unused",
        [["a", "b"]],
        "direct",
        layer=0,
        device="cpu",
        prompt_template="%s",
        callbacks=[Callback()],
        save_dir=tmp_path / "adapter",
        output_dir=str(tmp_path / "trainer"),
        max_steps=1,
    )
    assert returned_tokenizer is tokenizer
    assert steps == [1]
    loaded, loaded_tokenizer = api.load("unused", tmp_path / "adapter", device="cpu")
    assert loaded_tokenizer is tokenizer
    torch.testing.assert_close(loaded(**batch()).logits, wrapper(**batch()).logits)


def test_real_trainer_optimizes_saves_and_predicts(tmp_path):
    from transformers import TrainerCallback, TrainingArguments

    from easysteer.training.api import SteeringTrainer

    wrapper = SteeringModel(tiny_model(), TrainingConfig(algorithm="direct", layer=0))
    dataset = SupervisedDataset(Tokenizer(), [["a", "bc"], ["de", "f"]], "%s")
    events = []

    class Callback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            events.append(state.global_step)

    trainer = SteeringTrainer(
        model=wrapper,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            max_steps=2,
            per_device_train_batch_size=2,
            learning_rate=0.01,
            report_to=[],
            save_strategy="steps",
            save_steps=1,
            remove_unused_columns=False,
            label_names=["labels"],
        ),
        train_dataset=dataset,
        data_collator=SupervisedCollator(0),
        callbacks=[Callback()],
    )
    trainer.train()
    assert events == [1, 2]
    assert wrapper.adapter.bias.abs().sum() > 0
    assert (tmp_path / "checkpoint-2" / "steering_adapter.json").exists()
    result = trainer.predict(dataset)
    assert result.predictions.shape == (2, 5, 32)
    assert torch.isfinite(torch.tensor(result.metrics["test_loss"]))
    with pytest.raises(ValueError, match="resume is unsupported"):
        trainer.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))


@pytest.mark.parametrize("use_cache", [False, True])
def test_generation_matches_explicit_last_prompt_steering(use_cache):
    model = tiny_model()
    wrapper = SteeringModel(model, TrainingConfig(algorithm="direct", layer=0))
    wrapper.adapter.bias.data.fill_(0.3)
    prompt = torch.tensor([[1, 4, 5]])
    result = wrapper.generate(
        prompt, max_new_tokens=3, do_sample=False, use_cache=use_cache
    )
    expected = prompt
    for _ in range(result.shape[1] - prompt.shape[1]):
        logits = wrapper(
            input_ids=expected, steering_positions=torch.tensor([2]), use_cache=False
        ).logits
        expected = torch.cat(
            [expected, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1
        )
    torch.testing.assert_close(result, expected)
    assert not model.model.layers[0]._forward_hooks


@pytest.mark.parametrize("mask", [[[1, 1, 0]], [[0, 0, 0]], [[1, 0, 1]]])
def test_generation_rejects_right_padding_and_empty_prompts(mask):
    wrapper = SteeringModel(tiny_model(), TrainingConfig(layer=0))
    with pytest.raises(ValueError, match="left-padded"):
        wrapper.generate(
            torch.tensor([[1, 4, 0]]),
            attention_mask=torch.tensor(mask),
            max_new_tokens=1,
        )


def test_checkpoint_import_does_not_load_torch_or_transformers():
    code = """
import sys
from easysteer.training import load_checkpoint, TrainingConfig
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert not any('pyreft' in name or 'pyvene' in name for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
