"""Shared model loading, training and inference for ReFT interventions."""

import os

import torch
import transformers

from easysteer.reft import pyreft
from easysteer.reft.pyreft.core.modeling.modeling_utils import (
    get_dimension_by_component,
)
from easysteer.reft.pyreft.reft.algorithms import BiasIntervention

# Default Qwen chat format; callers can supply another prompt template.
PROMPT_TEMPLATE = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"

DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 100.0,
    "output_dir": "./tmp",
    "per_device_train_batch_size": 10,
    "learning_rate": 4e-3,
    "logging_steps": 40,
    "report_to": [],
    "save_strategy": "no",
}


def resolve_model_path(model_path=None):
    """Return the model path, falling back to $REFT_MODEL_PATH.

    Raises:
        ValueError: If neither the argument nor the environment variable
            is set.
    """
    path = model_path or os.environ.get("REFT_MODEL_PATH")
    if not path:
        raise ValueError(
            "No model path given: pass model_path=... or set the "
            "REFT_MODEL_PATH environment variable to the HF model directory"
        )
    return path


def load_model_and_tokenizer(model_path=None, device="cuda"):
    """Load the causal LM (bfloat16) and its tokenizer."""
    model_path = resolve_model_path(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, model_max_length=2048, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _build_intervention(intervention, embed_dim, low_rank_dimension):
    if intervention == "loreft":
        return pyreft.LoreftIntervention(
            embed_dim=embed_dim, low_rank_dimension=low_rank_dimension
        )
    if intervention == "bias":
        return BiasIntervention(embed_dim=embed_dim)
    raise ValueError(
        f"Unknown intervention '{intervention}'; expected 'loreft' or 'bias'"
    )


def _component_dimension(model, component):
    """Resolve a sequence component's width for the last-position helper."""
    if component.startswith("head_"):
        raise ValueError(
            "train_reft selects token positions, not heads; use the lower-level "
            "ReftConfig and unit_locations APIs for head interventions"
        )
    dimension = get_dimension_by_component(type(model), model.config, component)
    if dimension is None:
        raise ValueError(
            f"No dimension is registered for component {component!r} on "
            f"{type(model).__name__}; use a lower-level intervention with "
            "an explicit embed_dim for custom module paths"
        )
    return dimension


def train_reft(
    model_path,
    examples,
    intervention="loreft",
    *,
    layer=8,
    component="block_output",
    low_rank_dimension=4,
    device="cuda",
    prompt_template=PROMPT_TEMPLATE,
    callbacks=None,
    save_dir=None,
    **training_args,
):
    """Train a ReFT intervention on last-position supervised examples.

    Args:
        model_path: Model to fine-tune (falls back to $REFT_MODEL_PATH).
        examples: List of [instruction, response] pairs.
        intervention: "loreft" or "bias".
        layer: Layer whose representation is intervened.
        component: Registered sequence component (e.g. "block_output"). Head
            units require the lower-level API. The vLLM checkpoint adapter
            currently accepts only "block_output".
        low_rank_dimension: LoReFT rank (ignored for "bias").
        device: Device for training.
        prompt_template: Template applied to each instruction.
        callbacks: Optional list of transformers TrainerCallback objects.
        save_dir: If set, save the trained intervention there.
        **training_args: Overrides for transformers.TrainingArguments
            (see DEFAULT_TRAINING_ARGS for the defaults).

    Returns:
        The trained ReFT model on ``device`` and its tokenizer.
    """
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    representations = {
        "layer": layer,
        "component": component,
        "intervention": _build_intervention(
            intervention, _component_dimension(model, component), low_rank_dimension
        ),
    }
    if intervention == "loreft":
        representations["low_rank_dimension"] = low_rank_dimension
    reft_config = pyreft.ReftConfig(
        representations=representations,
        easysteer_training={
            "act_fn": "linear",
            "prompt_template": prompt_template,
            "apply": {"prompt_positions": [-1]},
        },
    )
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(device)
    reft_model.print_trainable_parameters()

    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer,
        model,
        [prompt_template % example[0] for example in examples],
        [example[1] for example in examples],
    )

    args = dict(DEFAULT_TRAINING_ARGS)
    args.update(training_args)
    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        args=transformers.TrainingArguments(**args),
        processing_class=tokenizer,
        **data_module,
    )
    for callback in callbacks or []:
        trainer.add_callback(callback)
    trainer.train()

    if save_dir is not None:
        reft_model.set_device("cpu")
        reft_model.save(save_directory=save_dir, save_to_hf_hub=False)
        reft_model.set_device(device)
    return reft_model, tokenizer


def load_reft(model_path, save_dir, device="cuda"):
    """Load a saved ReFT intervention on top of a freshly loaded base model.

    Returns:
        The loaded ReFT model and its tokenizer.
    """
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    reft_model = pyreft.ReftModel.load(save_dir, model)
    reft_model.set_device(device)
    return reft_model, tokenizer


def generate_reft(
    reft_model,
    tokenizer,
    instruction,
    *,
    device="cuda",
    max_new_tokens=512,
    prompt_template=PROMPT_TEMPLATE,
):
    """Generate a response with the intervention applied at the last prompt token."""
    prompt = tokenizer(prompt_template % instruction, return_tensors="pt").to(device)
    last_position = prompt["input_ids"].shape[-1] - 1
    _, response = reft_model.generate(
        prompt,
        unit_locations={"sources->base": (None, [[[last_position]]])},
        intervene_on_prompt=True,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
    )
    return tokenizer.decode(response[0], skip_special_tokens=True)
