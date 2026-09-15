# SPDX-License-Identifier: Apache-2.0
"""Adapt padded HF token batches to vLLM's shared selection runtime."""

import torch
from vllm.model_hooks.selection.batch import BatchView
from vllm.model_hooks.selection.runtime import collect_positions_apply_spec
from vllm.model_hooks.selection.spec import SelectSpec


def valid_token_mask(input_ids, attention_mask=None):
    """Validate a full sequence or cached suffix and return its binary mask."""
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or min(input_ids.shape) == 0
        or input_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("input_ids must be a nonempty [batch, tokens] integer tensor")
    if attention_mask is None:
        return torch.ones_like(input_ids, dtype=torch.bool)
    if (
        not isinstance(attention_mask, torch.Tensor)
        or attention_mask.ndim != 2
        or attention_mask.shape[0] != input_ids.shape[0]
        or attention_mask.shape[1] < input_ids.shape[1]
        or ((attention_mask != 0) & (attention_mask != 1)).any()
    ):
        raise ValueError("attention_mask must be binary and cover every input token")
    mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    if not mask.any(dim=1).all():
        raise ValueError("every sequence must contain a nonempty prompt")
    # Left and right padding are both valid for teacher forcing. Holes are not:
    # compacting them would no longer describe a contiguous causal sequence.
    starts = (mask[:, 1:] & ~mask[:, :-1]).sum(dim=1) + mask[:, 0]
    if (starts != 1).any():
        raise ValueError("attention_mask must contain one contiguous token sequence")
    return mask


def collect_training_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_lengths: torch.Tensor,
    apply: SelectSpec,
) -> torch.Tensor:
    """Return selected indices in the flattened, padded decoder output.

    A cached forward supplies suffix token IDs and the full attention mask.
    Prompt lengths always count real tokens and stay fixed during generation.
    ``num_output=0`` makes the shared resolver derive each generation index
    from its absolute token position, including multi-token teacher forcing
    and uncached recomputation, instead of one decode index per request.
    """
    mask = valid_token_mask(input_ids, attention_mask)
    if (
        not isinstance(prompt_lengths, torch.Tensor)
        or prompt_lengths.shape != (input_ids.shape[0],)
        or prompt_lengths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("prompt_lengths must contain one integer per batch row")
    prompt_lengths = prompt_lengths.to(device=input_ids.device, dtype=torch.long)
    total_lengths = mask.sum(dim=1)
    if ((prompt_lengths < 1) | (prompt_lengths > total_lengths)).any():
        raise ValueError("prompt_lengths must identify a nonempty prompt in each row")
    current_mask = mask[:, -input_ids.shape[1] :]
    counts = current_mask.sum(dim=1)
    batch = BatchView(
        query_start_loc=torch.cat([counts.new_zeros(1), counts.cumsum(0)]),
        num_computed=total_lengths - counts,
        num_prompt=prompt_lengths,
        num_output=torch.zeros_like(prompt_lengths),
    )
    padded_indices = current_mask.flatten().nonzero().flatten()
    selected = collect_positions_apply_spec(
        input_ids.flatten()[padded_indices], batch, apply.to_wire()
    )
    if selected is None:
        return padded_indices[:0]
    return padded_indices[selected]
