# SPDX-License-Identifier: Apache-2.0
"""Training and inference share token selection across padding and decoding."""

import copy
import json

import pytest
import torch
from test_training import batch, tiny_model
from vllm.model_hooks.selection.batch import BatchView
from vllm.model_hooks.selection.runtime import collect_positions_apply_spec
from vllm.model_hooks.selection.spec import SelectSpec
from vllm.model_hooks.steering.api import ApplySpec

from easysteer.training import TrainingConfig, load_checkpoint
from easysteer.training.model import SteeringModel
from easysteer.training.selection import collect_training_positions


@pytest.mark.parametrize(
    "clause,expected",
    [
        ({"prompt_positions": [-1, 99]}, [4, 9]),
        ({"prompt_window": (-3, -1)}, [2, 3, 8]),
        (
            {
                "prompt": "all",
                "exclude_prompt_tokens": [11, 21],
                "exclude_prompt_positions": [0],
            },
            [3, 4],
        ),
        (
            {
                "prompt_tokens": [10, 21],
                "generation_tokens": [15, 24],
                "exclude_generation_positions": [1],
            },
            [1, 9, 12],
        ),
        ({"generation_positions": [0, 2]}, [5, 7, 10, 12]),
        (
            {
                "generation_window": (0, 2),
                "exclude_generation_tokens": [14, 23],
            },
            [6, 10],
        ),
        ({"prompt_positions": [-100], "generation_window": (99, None)}, []),
        (
            {
                "prompt": "all",
                "generation": "all",
                "exclude_prompt_window": (-1, None),
                "exclude_generation_window": (1, None),
            },
            [1, 2, 3, 5, 8, 10],
        ),
    ],
)
def test_padded_teacher_forcing_uses_shared_selection(clause, expected):
    tokens = torch.tensor(
        [[0, 10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 0, 0, 0]]
    )
    mask = tokens != 0
    result = collect_training_positions(
        tokens, mask, torch.tensor([4, 2]), ApplySpec(**clause)
    )
    assert result.tolist() == expected


def test_teacher_forcing_and_vllm_decode_counts_select_same_tokens():
    tokens = torch.tensor([[1, 4, 5, 6, 7, 8]])
    apply = ApplySpec(
        prompt_positions=[-1], generation_window=(0, 3), exclude_generation_tokens=[7]
    )
    full = collect_training_positions(tokens, None, torch.tensor([3]), apply).tolist()
    selected = []
    for position in range(tokens.shape[1]):
        # This is the engine's actual per-request decode-step convention.
        geometry = BatchView(
            query_start_loc=torch.tensor([0, 1]),
            num_computed=torch.tensor([position]),
            num_prompt=torch.tensor([3]),
            num_output=torch.tensor([max(0, position - 3 + 1)]),
        )
        result = collect_positions_apply_spec(
            tokens[0, position : position + 1], geometry, apply.to_wire()
        )
        if result is not None:
            selected.append(position)
    assert full == selected == [2, 3, 5]


@pytest.mark.parametrize("algorithm", ["direct", "loreft"])
def test_nonmatching_batch_retains_zero_gradients_for_all_adapter_parameters(algorithm):
    wrapper = SteeringModel(
        tiny_model(),
        TrainingConfig(
            algorithm=algorithm, layer=0, rank=2, apply={"prompt_tokens": [999]}
        ),
    )
    values = batch()
    baseline = wrapper.base_model(
        input_ids=values["input_ids"],
        attention_mask=values["attention_mask"],
        labels=values["labels"],
    )
    output = wrapper(**values)
    torch.testing.assert_close(output.logits, baseline.logits)
    output.loss.backward()
    assert all(
        parameter.grad is not None and not parameter.grad.any()
        for parameter in wrapper.adapter.parameters()
    )


def test_custom_policy_round_trip_and_v1_checkpoint_compatibility(tmp_path):
    apply = SelectSpec(
        prompt_window=(-3, None),
        generation_positions=[0, 2],
        exclude_prompt_tokens=[4],
        exclude_generation_window=(1, 2),
    )
    config = TrainingConfig(algorithm="direct", layer=0, apply=apply)
    original = copy.deepcopy(config.apply)
    apply.prompt_window = (0, 1)
    assert config.apply == original
    wrapper = SteeringModel(tiny_model(), config)
    target = wrapper.save(tmp_path)
    checkpoint = load_checkpoint(target)
    assert checkpoint.config.apply == original
    spec = checkpoint.to_spec()
    assert spec.vectors[0].apply == original
    spec.vectors[0].apply.prompt_window = (0, 1)
    assert checkpoint.config.apply == original
    data = json.loads(target.read_text())
    assert data["version"] == 2
    data["version"] = 1
    data["config"]["apply"] = {"prompt_positions": [-1]}
    target.write_text(json.dumps(data))
    assert load_checkpoint(target).config.apply == ApplySpec(prompt_positions=[-1])


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("num_beams", [1, 2])
def test_generation_policy_matches_recomputation_for_padded_batches(
    use_cache, num_beams
):
    torch.manual_seed(11)
    wrapper = SteeringModel(
        tiny_model(),
        TrainingConfig(
            algorithm="direct",
            layer=0,
            apply=ApplySpec(
                prompt_window=(-2, None),
                generation="all",
                exclude_generation_positions=[1],
            ),
        ),
    )
    wrapper.adapter.bias.data.copy_(torch.linspace(-1, 1, 16))
    prompt = torch.tensor([[1, 4, 5], [0, 1, 7]])
    mask = prompt != 0
    lengths = mask.sum(dim=1)
    observed = []

    def record(module, args, kwargs, output):
        tokens = kwargs["input_ids"]
        observed.append(
            (
                tokens.detach().clone(),
                kwargs["attention_mask"].detach().clone(),
                output.logits.detach().clone(),
            )
        )

    handle = wrapper.base_model.register_forward_hook(record, with_kwargs=True)
    try:
        generated = wrapper.generate(
            prompt,
            attention_mask=mask,
            max_new_tokens=4,
            min_new_tokens=4,
            do_sample=False,
            use_cache=use_cache,
            num_beams=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
        )
    finally:
        handle.remove()
    # Compare every cached forward with a full recomputation of the same
    # generated prefix. Beam reorderings are resolved from the returned token
    # histories by comparing whole generation outputs below.
    if num_beams == 1:
        history = prompt
        for index, (tokens, full_mask, logits) in enumerate(observed):
            if index:
                history = torch.cat([history, tokens[:, -1:]], dim=1)
            position_ids = full_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(~full_mask.bool(), 0)
            expected = wrapper(
                history,
                attention_mask=full_mask,
                prompt_lengths=lengths,
                position_ids=position_ids,
                use_cache=False,
            ).logits[:, -1]
            torch.testing.assert_close(logits[:, -1], expected, rtol=1e-4, atol=1e-5)
    other = wrapper.generate(
        prompt,
        attention_mask=mask,
        max_new_tokens=4,
        min_new_tokens=4,
        do_sample=False,
        use_cache=not use_cache,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_scores=True,
    )
    torch.testing.assert_close(generated.sequences, other.sequences)
    for actual, expected in zip(generated.scores, other.scores):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert not wrapper.base_model._forward_pre_hooks
    assert not wrapper.base_model.model.layers[0]._forward_hooks


@pytest.mark.parametrize("mask", [[[1, 0, 1, 1]], [[0, 0, 0, 0]], [[1, 2, 1, 1]]])
def test_invalid_mask_rejected_before_installing_hooks(mask):
    wrapper = SteeringModel(tiny_model(), TrainingConfig(layer=0))
    with pytest.raises(ValueError, match="attention_mask|nonempty"):
        wrapper(
            torch.tensor([[1, 4, 5, 6]]),
            attention_mask=torch.tensor(mask),
            prompt_lengths=torch.tensor([2]),
        )
    assert not wrapper.base_model.model.layers[0]._forward_hooks


def test_unrecorded_position_override_is_rejected():
    wrapper = SteeringModel(tiny_model(), TrainingConfig(layer=0))
    with pytest.raises(ValueError, match="TrainingConfig.apply"):
        wrapper(**batch(), steering_positions=torch.tensor([1, 0]))


@pytest.mark.parametrize("source", ["argument", "generation_config", "model_default"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_static_cache_rejected_only_when_enabled(monkeypatch, source, use_cache):
    from transformers import GenerationConfig

    wrapper = SteeringModel(tiny_model(), TrainingConfig(layer=0))
    kwargs = {"max_new_tokens": 1, "use_cache": use_cache}
    if source == "argument":
        kwargs["cache_implementation"] = "static"
    elif source == "generation_config":
        kwargs["generation_config"] = GenerationConfig(cache_implementation="static")
    else:
        wrapper.base_model.generation_config.cache_implementation = "static"
    prompt = torch.tensor([[1, 4, 5]])
    if use_cache:
        monkeypatch.setattr(
            wrapper.base_model,
            "generate",
            lambda *args, **kwargs: pytest.fail("entered HF generate"),
        )
        with pytest.raises(ValueError, match="does not support static caches"):
            wrapper.generate(prompt, **kwargs)
    else:
        output = wrapper.generate(prompt, **kwargs)
        assert output.shape == (1, prompt.shape[1] + 1)
    assert not wrapper.base_model._forward_pre_hooks
    assert not wrapper.base_model.model.layers[0]._forward_hooks
    assert wrapper.training


def test_generation_config_can_disable_inherited_static_cache():
    from transformers import GenerationConfig

    wrapper = SteeringModel(tiny_model(), TrainingConfig(layer=0))
    config = GenerationConfig(cache_implementation="static", use_cache=False)
    prompt = torch.tensor([[1, 4, 5]])
    output = wrapper.generate(prompt, generation_config=config, max_new_tokens=1)
    assert output.shape == (1, prompt.shape[1] + 1)
