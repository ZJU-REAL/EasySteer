# SPDX-License-Identifier: Apache-2.0
"""Public frozen-model training, checkpoint restoration, and generation."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .checkpoint import PROMPT_TEMPLATE, TrainingConfig, load_checkpoint
from .data import SupervisedCollator, SupervisedDataset
from .model import SteeringModel

DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 100.0,
    "output_dir": "./tmp",
    "per_device_train_batch_size": 10,
    "learning_rate": 4e-3,
    "logging_steps": 40,
    "report_to": [],
    "save_strategy": "no",
    "remove_unused_columns": False,
    "label_names": ["labels"],
}


def load_model_and_tokenizer(model_path=None, device="cuda"):
    model_path = model_path or os.environ.get("EASYSTEER_MODEL_PATH")
    if not model_path:
        raise ValueError("pass model_path or set EASYSTEER_MODEL_PATH")
    device = torch.device(device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class SteeringTrainer(Trainer):
    """Use standard Transformers optimization while saving only adapter weights."""

    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save:
            self.model.save(output_dir or self.args.output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        raise ValueError(
            "native checkpoints store the materialized rotation, not its training parameterization; "
            "exact Trainer resume is unsupported"
        )

    def _load_best_model(self):
        self.model.load_adapter(load_checkpoint(self.state.best_model_checkpoint))


def train(
    model_path,
    examples,
    algorithm="loreft",
    *,
    layer=8,
    component="hidden_states",
    rank=4,
    device="cuda",
    prompt_template=PROMPT_TEMPLATE,
    callbacks=None,
    save_dir=None,
    max_length=2048,
    **training_args,
):
    """Train direct or LoReFT steering at each example's last prompt token.

    Examples are instruction/response pairs. Loss covers response tokens only;
    responses may be truncated to max_length after preserving the whole prompt.
    Additional arguments configure transformers.TrainingArguments. Gradient
    checkpointing is unsupported because it replays module hooks in backward.
    """
    config = TrainingConfig(algorithm, component, layer, rank, prompt_template)
    if training_args.get("gradient_checkpointing"):
        raise ValueError("steering training does not support gradient checkpointing")
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    wrapper = SteeringModel(model, config)
    dataset = SupervisedDataset(tokenizer, examples, prompt_template, max_length)
    args = {**DEFAULT_TRAINING_ARGS, **training_args}
    args.setdefault("use_cpu", torch.device(device).type == "cpu")
    trainer = SteeringTrainer(
        model=wrapper,
        args=TrainingArguments(**args),
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=SupervisedCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer.train()
    if save_dir is not None:
        wrapper.save(save_dir)
    return wrapper, tokenizer


def load(model_path, save_dir, device="cuda"):
    """Restore a native adapter on a freshly loaded, frozen base model."""
    checkpoint = load_checkpoint(save_dir)
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    wrapper = SteeringModel(model, checkpoint.config)
    wrapper.load_adapter(checkpoint)
    return wrapper, tokenizer


def generate(
    model,
    tokenizer,
    instruction,
    *,
    device=None,
    max_new_tokens=512,
    prompt_template=None,
    **generation_args,
):
    """Generate with the checkpoint's prompt format and last-prompt selection."""
    template = prompt_template or model.training_config.prompt_template
    device = device or next(model.base_model.parameters()).device
    inputs = tokenizer(template % instruction, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, **generation_args)
    return tokenizer.decode(output[0], skip_special_tokens=True)
