# SPDX-License-Identifier: Apache-2.0
"""Differentiable transforms at the same decoder boundary used by vLLM."""

from contextlib import contextmanager

import torch
from torch import nn

from .checkpoint import SteeringCheckpoint, TrainingConfig
from .selection import collect_training_positions, valid_token_mask


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
    loss_type = "ForCausalLM"
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
    def _steering(self, positions):
        """Apply the adapter to selected flat rows; resolve once per forward."""
        target = self.base_model.get_submodule(self._target_name)

        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.ndim != 3
                or hidden.shape[-1] != self._width
            ):
                raise ValueError(
                    "decoder output must contain [batch, tokens, hidden_width]"
                )
            selected = positions() if callable(positions) else positions
            selected = selected.to(device=hidden.device, dtype=torch.long)
            flat = hidden.reshape(-1, self._width)
            # Keep the adapter in the autograd graph even when this batch has
            # no matching tokens. DDP then receives zero gradients on all ranks.
            transformed = self.adapter(flat.index_select(0, selected))
            result = flat.index_copy(0, selected, transformed).view_as(hidden)
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
        prompt_lengths=None,
        **kwargs,
    ):
        if "steering_positions" in kwargs:
            raise ValueError("set TrainingConfig.apply instead of steering_positions")
        if getattr(self.base_model, "is_gradient_checkpointing", False):
            raise ValueError(
                "steering training does not support gradient checkpointing"
            )
        positions = collect_training_positions(
            input_ids, attention_mask, prompt_lengths, self.training_config.apply
        )
        with self._steering(positions):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Apply the checkpoint's selection to prefill and every decode step.

        Prompts may be left padded. Dynamic caches and uncached full-sequence
        recomputation use the same absolute positions as teacher forcing.
        Static caches are unsupported because they replace the token mask
        needed for selection with a precomputed attention mask.
        Beam search and multiple returned sequences preserve the original
        request groups, so each expanded group retains its prompt length.
        """
        if any(
            kwargs.get(key) is not None
            for key in (
                "inputs_embeds",
                "past_key_values",
                "steering_positions",
                "prompt_lengths",
            )
        ):
            raise ValueError(
                "generation requires complete token prompts; cache, embedding, "
                "and position overrides are unsupported"
            )
        generation_config = (
            kwargs.get("generation_config") or self.base_model.generation_config
        )
        use_cache = kwargs.setdefault(
            "use_cache", getattr(generation_config, "use_cache", True)
        )
        cache_implementation = kwargs.get(
            "cache_implementation", generation_config.cache_implementation
        )
        if use_cache and cache_implementation in {"static", "offloaded_static"}:
            raise ValueError(
                "steering generation does not support static caches; "
                "use cache_implementation='dynamic' or use_cache=False"
            )
        mask = valid_token_mask(input_ids, attention_mask)
        if (
            mask.shape != input_ids.shape
            or not mask[:, -1].all()
            or (mask[:, :-1] & ~mask[:, 1:]).any()
        ):
            raise ValueError(
                "batched generation requires nonempty, left-padded prompts"
            )
        prompt_lengths = mask.sum(dim=1)
        positions = None

        def select(module, args, model_kwargs):
            nonlocal positions
            tokens = model_kwargs.get("input_ids")
            if tokens is None and args:
                tokens = args[0]
            if tokens is None:
                raise ValueError(
                    "generation selection requires input_ids on every forward"
                )
            if tokens.shape[0] % len(prompt_lengths):
                raise ValueError("generation changed the original request groups")
            lengths = prompt_lengths.repeat_interleave(
                tokens.shape[0] // len(prompt_lengths)
            )
            positions = collect_training_positions(
                tokens,
                model_kwargs.get("attention_mask"),
                lengths,
                self.training_config.apply,
            )

        was_training = self.training
        self.eval()
        handle = self.base_model.register_forward_pre_hook(select, with_kwargs=True)
        try:
            with self._steering(lambda: positions):
                return self.base_model.generate(
                    input_ids=input_ids, attention_mask=mask.long(), **kwargs
                )
        finally:
            handle.remove()
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
