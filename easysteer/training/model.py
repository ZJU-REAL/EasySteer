# SPDX-License-Identifier: Apache-2.0
"""Differentiable transforms at the same decoder boundary used by vLLM."""

from contextlib import contextmanager

import torch
from torch import nn

from .checkpoint import SteeringCheckpoint, TrainingConfig


class DirectSteering(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, hidden):
        return (hidden.float() + self.bias).to(hidden.dtype)


class LoReFTSteering(nn.Module):
    """LoReFT: h + (h W^T + b - h R) R^T, with orthonormal columns of R.

    A reduced QR keeps the trainable rotation linear in hidden width times
    rank. Checkpoints store the materialized basis used by vLLM inference.
    """

    def __init__(self, width: int, rank: int):
        super().__init__()
        if rank > width:
            raise ValueError("LoReFT rank cannot exceed the hidden width")
        self.rotation = nn.Parameter(torch.empty(width, rank))
        nn.init.orthogonal_(self.rotation)
        self.source = nn.Linear(width, rank)

    def basis(self):
        basis, upper = torch.linalg.qr(self.rotation.float(), mode="reduced")
        # Fix QR's sign ambiguity, including when reloading an exported basis.
        signs = torch.where(upper.diagonal() < 0, -1.0, 1.0)
        return basis * signs

    def forward(self, hidden):
        values = hidden.float()
        basis = self.basis()
        output = values + (self.source(values) - values @ basis) @ basis.T
        return output.to(hidden.dtype)


def _decoder_layer(model, layer):
    """Resolve ordinary HF decoder stacks without importing model families."""
    candidates = []
    for path in (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
    ):
        try:
            stack = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(stack, nn.ModuleList) and all(
            stack is not old for old in candidates
        ):
            candidates.append(stack)
    if len(candidates) != 1:
        raise ValueError("training requires one supported causal decoder stack")
    stack = candidates[0]
    if layer >= len(stack):
        raise ValueError(
            f"layer {layer} is outside the {len(stack)}-layer decoder stack"
        )
    return stack[layer]


class SteeringModel(nn.Module):
    """A frozen causal LM with one trainable hidden-state steering adapter."""

    accepts_loss_kwargs = False
    main_input_name = "input_ids"

    def __init__(self, model: nn.Module, config: TrainingConfig):
        super().__init__()
        if getattr(model, "is_gradient_checkpointing", False):
            raise ValueError(
                "steering training does not support gradient checkpointing"
            )
        target = _decoder_layer(model, config.layer)
        embeddings = model.get_input_embeddings()
        width = getattr(model.config, "hidden_size", None)
        if type(width) is not int or width < 1:
            raise ValueError("training requires a positive config.hidden_size")
        self.base_model = model.requires_grad_(False)
        self.config = model.config
        self.training_config = config
        self.adapter = (
            DirectSteering(width)
            if config.algorithm == "direct"
            else LoReFTSteering(width, config.rank)
        ).to(device=embeddings.weight.device)
        # Keep the target as a name, avoiding duplicate module/state registration.
        self._target_name = next(
            name for name, module in model.named_modules() if module is target
        )
        self._width = width
        self.base_model.eval()

    def train(self, mode=True):
        super().train(mode)
        # Frozen model dropout must not change the function being optimized.
        self.base_model.eval()
        return self

    @contextmanager
    def _steering(self, positions, *, cached_generation=False, expand_batch=False):
        if (
            not isinstance(positions, torch.Tensor)
            or positions.ndim != 1
            or positions.numel() == 0
            or positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("steering positions must be a nonempty integer vector")
        target = self.base_model.get_submodule(self._target_name)
        calls = 0

        def hook(module, args, output):
            nonlocal calls
            calls += 1
            if cached_generation and calls > 1:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.ndim != 3
                or hidden.shape[-1] != self._width
            ):
                raise ValueError(
                    "decoder output must contain [batch, tokens, hidden_width]"
                )
            selected = positions.to(device=hidden.device, dtype=torch.long)
            if hidden.shape[0] % len(selected) or (
                not expand_batch and hidden.shape[0] != len(selected)
            ):
                raise ValueError(
                    "steering positions must identify one token per batch row"
                )
            selected = selected.repeat_interleave(hidden.shape[0] // len(selected))
            if ((selected < 0) | (selected >= hidden.shape[1])).any():
                raise ValueError("steering position is outside the decoder sequence")
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            result = hidden.clone()
            result[rows, selected] = self.adapter(hidden[rows, selected])
            return (result, *output[1:]) if isinstance(output, tuple) else result

        handle = target.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        steering_positions=None,
        **kwargs,
    ):
        if steering_positions is None:
            raise ValueError("forward requires explicit steering_positions")
        with self._steering(steering_positions):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **kwargs):
        if input_ids.ndim != 2 or min(input_ids.shape) == 0:
            raise ValueError("generation input_ids must have shape [batch, tokens]")
        if attention_mask is None:
            positions = torch.full(
                (input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device
            )
        else:
            if (
                attention_mask.shape != input_ids.shape
                or ((attention_mask != 0) & (attention_mask != 1)).any()
            ):
                raise ValueError(
                    "generation requires a binary attention mask matching input_ids"
                )
            if (
                not attention_mask[:, -1].all()
                or (attention_mask[:, 1:] < attention_mask[:, :-1]).any()
            ):
                raise ValueError(
                    "batched generation requires nonempty, left-padded prompts"
                )
            tokens = torch.arange(input_ids.shape[1], device=input_ids.device)
            positions = torch.where(attention_mask.bool(), tokens, -1).max(dim=1).values
        use_cache = kwargs.setdefault("use_cache", True)
        was_training = self.training
        self.eval()
        try:
            with self._steering(
                positions, cached_generation=use_cache, expand_batch=True
            ):
                return self.base_model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, **kwargs
                )
        finally:
            self.train(was_training)

    def to_payload(self):
        from vllm.model_hooks.steering.payloads import DirectionVector, ReftIntervention

        if self.training_config.algorithm == "direct":
            return DirectionVector({self.training_config.layer: self.adapter.bias})
        return ReftIntervention(
            rotate_layer=self.adapter.basis(),
            learned_source_weight=self.adapter.source.weight,
            learned_source_bias=self.adapter.source.bias,
            layer=self.training_config.layer,
        )

    def to_spec(self):
        return SteeringCheckpoint(self.training_config, self.to_payload()).to_spec()

    def save(self, path):
        return SteeringCheckpoint(self.training_config, self.to_payload()).save(path)

    def load_adapter(self, checkpoint: SteeringCheckpoint):
        if checkpoint.config != self.training_config:
            raise ValueError("checkpoint config does not match the steering model")
        payload = checkpoint.payload
        with torch.no_grad():
            if self.training_config.algorithm == "direct":
                weights = [
                    (self.adapter.bias, payload.layers[self.training_config.layer])
                ]
            else:
                weights = [
                    (self.adapter.rotation, payload.rotate_layer),
                    (self.adapter.source.weight, payload.learned_source_weight),
                    (self.adapter.source.bias, payload.learned_source_bias),
                ]
            for parameter, value in weights:
                if value is None or tuple(parameter.shape) != tuple(value.shape):
                    raise ValueError(
                        "checkpoint dimensions do not match the base model"
                    )
            for parameter, value in weights:
                parameter.copy_(torch.tensor(value, device=parameter.device))
