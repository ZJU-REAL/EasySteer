"""CPU contracts for the vendored ReFT Trainer and its shared data collator."""

from types import SimpleNamespace

import pytest
import torch

from easysteer.reft.pyreft.data import ReftDataCollator
from easysteer.reft.pyreft.reft.trainer import (
    ReftTrainer,
    ReftTrainerForCausalLM,
    ReftTrainerForSequenceClassification,
)


def inputs():
    return {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
        "intervention_locations": torch.tensor([[[1]], [[0]]]),
    }


@pytest.mark.parametrize("counterfactual", [True, False])
@pytest.mark.parametrize("return_outputs", [True, False])
def test_compute_loss_returns_scalar_loss(counterfactual, return_outputs):
    output = SimpleNamespace(loss=torch.tensor(2.0, requires_grad=True))
    calls = []

    def intervenable(base, **kwargs):
        calls.append((base, kwargs))
        return (None, output) if counterfactual else (output, None)

    result = ReftTrainer.compute_loss(
        None,
        intervenable,
        inputs(),
        return_outputs=return_outputs,
        num_items_in_batch=torch.tensor(2),
    )
    loss = result[0] if return_outputs else result
    assert loss is output.loss
    assert loss.ndim == 0 and loss.requires_grad
    if return_outputs:
        assert result[1] is output
    assert calls[0][1]["unit_locations"] == {"sources->base": (None, [[[1], [0]]])}
    assert "num_items_in_batch" not in calls[0][1]


def test_classification_accepts_trainer_loss_kwargs():
    logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]], requires_grad=True)
    output = SimpleNamespace(logits=logits)
    model = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(problem_type=None), num_labels=2)
    )
    trainer = SimpleNamespace(model=model)
    batch = inputs()
    loss, result = ReftTrainerForSequenceClassification.compute_loss(
        trainer,
        lambda *args, **kwargs: (None, output),
        batch,
        return_outputs=True,
        num_items_in_batch=torch.tensor(2),
    )
    torch.testing.assert_close(
        loss, torch.nn.functional.cross_entropy(logits, batch["labels"])
    )
    assert result is output
    loss.backward()
    assert logits.grad is not None


def test_data_collator_preserves_batch_and_clips_locations():
    locations = torch.tensor([[[0, 1, 2, 3]], [[1, 0, 3, 2]]])
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "intervention_locations": locations,
        "labels": torch.tensor([0, 1]),
    }
    result = ReftDataCollator(lambda _: dict(batch))([{}, {}])
    assert result["input_ids"] is batch["input_ids"]
    assert result["labels"] is batch["labels"]
    torch.testing.assert_close(result["intervention_locations"], locations[..., :2])


def test_tiny_causal_model_backward_and_trainer_prediction(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments

    from easysteer.reft import pyreft
    from easysteer.reft.pyreft.reft.algorithms import BiasIntervention

    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            use_cache=False,
        )
    )
    intervention = BiasIntervention(embed_dim=16)
    wrapper = pyreft.get_reft_model(
        model,
        pyreft.ReftConfig(
            representations={
                "layer": 0,
                "component": "block_output",
                "intervention": intervention,
            }
        ),
    )
    trainer = ReftTrainerForCausalLM(
        model=wrapper,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            label_names=["labels"],
            remove_unused_columns=False,
        ),
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "intervention_locations": torch.tensor([[[1]]]),
    }
    loss = trainer.compute_loss(wrapper, batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert intervention.bias.grad is not None
    assert torch.isfinite(intervention.bias.grad).all()
    eval_loss, logits, labels = trainer.prediction_step(
        wrapper,
        batch,
        prediction_loss_only=False,
    )
    assert eval_loss.ndim == 0 and torch.isfinite(eval_loss)
    assert logits.shape == (1, 4, 32)
    torch.testing.assert_close(labels, batch["labels"])
