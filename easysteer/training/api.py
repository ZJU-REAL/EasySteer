# SPDX-License-Identifier: Apache-2.0
"""Public frozen-model training, checkpoint restoration, and generation."""

import os
from functools import partial

import torch
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

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
    "average_tokens_across_devices": True,
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


def _response_loss(outputs, labels, num_items_in_batch=None, *, label_smoothing=0.0):
    # Trainer supplies the token count across all accumulated microbatches and
    # DDP ranks, and compensates for DDP's gradient averaging in compute_loss.
    logits = outputs.logits[..., :-1, :].float().contiguous()
    labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
        reduction="sum",
        label_smoothing=label_smoothing,
    )
    denominator = (
        labels.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    )
    return loss / denominator


class SteeringTrainer(Trainer):
    """Normalize response loss globally and save only adapter weights."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("compute_loss_func", _response_loss)
        super().__init__(*args, **kwargs)
        if not self.args.average_tokens_across_devices:
            raise ValueError("steering training requires average_tokens_across_devices")
        if self.compute_loss_func is _response_loss:
            self.compute_loss_func = partial(
                _response_loss, label_smoothing=self.args.label_smoothing_factor
            )

    def save_model(self, output_dir=None, _internal_call=False):
        if self.is_world_process_zero():
            self.model.save(output_dir or self.args.output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        raise ValueError(
            "native checkpoints store the materialized rotation, not its training parameterization; "
            "exact Trainer resume is unsupported"
        )

    def _load_best_model(self):
        self.model.load_adapter(load_checkpoint(self.state.best_model_checkpoint))


def _training_arguments(device, training_args):
    """Initialize the process's device before allocating the frozen model."""
    for name in ("fsdp", "deepspeed", "parallelism_config"):
        if training_args.get(name):
            raise ValueError(f"steering training supports DDP, not {name}")
    for name in ("tp_size", "tensor_parallel_size"):
        if name in training_args:
            raise ValueError("steering training does not support tensor parallelism")
    for name in ("ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"):
        if os.environ.get(name, "false").lower() in {"true", "1"}:
            raise ValueError(f"steering training supports DDP; unset {name}")

    requested = torch.device(device)
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("steering training supports CPU or CUDA devices")
    values = {**DEFAULT_TRAINING_ARGS, **training_args}
    values.setdefault("use_cpu", requested.type == "cpu")
    if values["use_cpu"] != (requested.type == "cpu"):
        raise ValueError("device and use_cpu select different device types")
    args = TrainingArguments(**values)
    actual = args.device
    if actual.type != requested.type:
        raise ValueError(f"requested {requested.type}, but Trainer selected {actual}")
    if requested.index is not None and requested.index != actual.index:
        raise ValueError(
            f"Trainer selected {actual}, which does not match device={device!r}; "
            "use device='cuda' with torchrun or CUDA_VISIBLE_DEVICES"
        )
    if args.n_gpu > 1:
        raise ValueError(
            "multiple visible GPUs require torchrun for DDP; select one GPU with "
            "CUDA_VISIBLE_DEVICES for single-device training"
        )
    return args


def train(
    model_path,
    examples,
    algorithm="loreft",
    *,
    layer=8,
    component="hidden_states",
    rank=4,
    apply=None,
    device="cuda",
    prompt_template=PROMPT_TEMPLATE,
    callbacks=None,
    save_dir=None,
    max_length=2048,
    **training_args,
):
    """Train direct or LoReFT steering with a vLLM token selection.

    Examples are instruction/response pairs. Loss covers response tokens only;
    responses may be truncated to max_length after preserving the whole prompt.
    Additional arguments configure transformers.TrainingArguments. Gradient
    checkpointing is unsupported because it replays module hooks in backward.
    Launch with torchrun for DDP; every rank holds the complete frozen model.
    By default, steering selects the last prompt token.
    """
    config = TrainingConfig(
        algorithm, component, layer, rank, prompt_template, apply=apply
    )
    if training_args.get("gradient_checkpointing"):
        raise ValueError("steering training does not support gradient checkpointing")
    args = _training_arguments(device, training_args)
    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(model_path, args.device)
    wrapper = SteeringModel(model, config)
    dataset = SupervisedDataset(tokenizer, examples, prompt_template, max_length)
    trainer = SteeringTrainer(
        model=wrapper,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=SupervisedCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer.train()
    if save_dir is not None:
        trainer.save_model(save_dir)
    trainer.accelerator.wait_for_everyone()
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
    """Generate with the checkpoint's prompt format and token selection."""
    template = prompt_template or model.training_config.prompt_template
    device = device or next(model.base_model.parameters()).device
    inputs = tokenizer(template % instruction, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, **generation_args)
    return tokenizer.decode(output[0], skip_special_tokens=True)
