# SPDX-License-Identifier: Apache-2.0
"""Verbatim copy of the pre-refactor position collector (commit d3b77a3).

Reference oracle for scripts/test_mask_equivalence.py ONLY — do not use
in production code. Kept so the mask-algebra rewrite can be fuzz-checked
for exact behavioral equivalence at any time.
"""

from typing import Dict, Optional

import torch


# ========== GPU-Optimized Helper Functions ==========


def get_decode_mask_gpu(
    current_tokens: torch.Tensor,
    sample_ids: torch.Tensor,
    is_decode_mask: torch.Tensor,
    generate_trigger_tokens: Optional[set],
    device: torch.device
) -> torch.Tensor:
    """
    Generate mask for decode sample positions using GPU batch operations.
    
    Args:
        current_tokens: [total_tokens] token IDs
        sample_ids: [total_tokens] sample ID for each token
        is_decode_mask: [num_samples] boolean tensor (True for decode samples)
        generate_trigger_tokens: Set of trigger token IDs for generation phase, or None
        device: Device to create tensors on
        
    Returns:
        mask: [total_tokens] boolean tensor indicating positions to apply intervention
    """
    # Get decode sample IDs
    decode_sample_ids = torch.nonzero(is_decode_mask, as_tuple=False).squeeze(-1)
    
    if decode_sample_ids.numel() == 0:
        return torch.zeros(len(sample_ids), dtype=torch.bool, device=device)
    
    # Apply to all decode tokens if configured (-1 means apply to all)
    if generate_trigger_tokens is not None and -1 in generate_trigger_tokens:
        mask = torch.isin(sample_ids, decode_sample_ids)
        return mask
    
    # Apply to specific decode tokens based on token IDs
    trigger_tensor = torch.tensor(
        list(generate_trigger_tokens),
        dtype=current_tokens.dtype,
        device=device
    )
    token_match = torch.isin(current_tokens, trigger_tensor)
    sample_match = torch.isin(sample_ids, decode_sample_ids)
    
    return token_match & sample_match


def apply_exclusions_gpu(
    mask: torch.Tensor,
    current_tokens: torch.Tensor,
    sample_ids: torch.Tensor,
    relative_positions: torch.Tensor,
    num_computed: Optional[torch.Tensor],
    prefill_sample_ids: torch.Tensor,
    prefill_exclude_positions: Optional[list],
    prefill_exclude_tokens: Optional[set],
    query_start_loc: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Apply exclusion rules using GPU batch operations.
    
    Args:
        mask: [total_tokens] boolean tensor to apply exclusions to
        current_tokens: [total_tokens] token IDs
        sample_ids: [total_tokens] sample ID for each token
        relative_positions: [total_tokens] relative position within each sample
        num_computed: [num_samples] cached token counts or None
        prefill_sample_ids: [num_prefill_samples] IDs of prefill samples
        prefill_exclude_positions: List of positions to exclude or None (supports negative indices)
        prefill_exclude_tokens: Set of token IDs to exclude or None
        query_start_loc: [num_samples+1] sample boundaries
        device: Device to create tensors on
        
    Returns:
        mask: Updated boolean tensor with exclusions applied
    """
    # ========== 1. Exclude by position ==========
    if prefill_exclude_positions is not None:
        # Separate positive and negative indices
        positive_excludes = [p for p in prefill_exclude_positions if p >= 0]
        negative_excludes = [p for p in prefill_exclude_positions if p < 0]
        
        # Compute original positions (accounting for cache)
        if num_computed is not None:
            num_computed_expanded = num_computed[sample_ids]
            original_positions = relative_positions + num_computed_expanded
        else:
            original_positions = relative_positions
        
        # ===== Handle positive indices =====
        if positive_excludes:
            exclude_positions_tensor = torch.tensor(
                positive_excludes,
                dtype=original_positions.dtype,
                device=device
            )
            exclude_position_match = torch.isin(original_positions, exclude_positions_tensor)
            
            # Only exclude from prefill samples
            exclude_position_match &= torch.isin(sample_ids, prefill_sample_ids)
            
            # Remove excluded positions
            mask &= ~exclude_position_match
        
        # ===== Handle negative indices (Python-style: -1 means last position) =====
        if negative_excludes:
            # Compute each sample's total length (including cached tokens)
            sample_lengths = query_start_loc[1:] - query_start_loc[:-1]  # [num_samples]
            if num_computed is not None:
                sample_total_lengths = sample_lengths + num_computed
            else:
                sample_total_lengths = sample_lengths
            
            # For each negative index
            for neg_idx in negative_excludes:
                # Convert negative index to absolute position for each sample
                # Example: length=19, neg_idx=-1 → abs_pos=18
                abs_positions_per_sample = sample_total_lengths + neg_idx  # [num_samples]
                
                # Expand to per-token: what's the target absolute position for this token's sample?
                abs_positions_expanded = abs_positions_per_sample[sample_ids]  # [total_tokens]
                
                # Match: is this token at the target absolute position?
                neg_exclude_match = (original_positions == abs_positions_expanded)
                
                # Only exclude from prefill samples
                neg_exclude_match &= torch.isin(sample_ids, prefill_sample_ids)
                
                # Remove excluded positions
                mask &= ~neg_exclude_match
    
    # ========== 2. Exclude by token ==========
    if prefill_exclude_tokens is not None:
        exclude_tokens_tensor = torch.tensor(
            list(prefill_exclude_tokens),
            dtype=current_tokens.dtype,
            device=device
        )
        exclude_token_match = torch.isin(current_tokens, exclude_tokens_tensor)
        
        # Remove excluded tokens
        mask &= ~exclude_token_match
    
    return mask


def get_prefill_mask_gpu(
    current_tokens: torch.Tensor,
    sample_ids: torch.Tensor,
    relative_positions: torch.Tensor,
    is_decode_mask: torch.Tensor,
    num_computed: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    prefill_trigger_tokens: Optional[set],
    prefill_trigger_positions: Optional[list],
    prefill_exclude_tokens: Optional[set],
    prefill_exclude_positions: Optional[list],
    device: torch.device
) -> torch.Tensor:
    """
    Generate mask for prefill sample positions using GPU batch operations.
    
    Handles:
    - Position-based triggers (prefill_trigger_positions)
    - Token-based triggers (prefill_trigger_tokens)
    - Exclusions (prefill_exclude_positions, prefill_exclude_tokens)
    - Prefix caching (num_computed)
    
    Args:
        current_tokens: [total_tokens] token IDs
        sample_ids: [total_tokens] sample ID for each token
        relative_positions: [total_tokens] relative position within each sample
        is_decode_mask: [num_samples] boolean tensor (True for decode samples)
        num_computed: [num_samples] cached token counts or None
        query_start_loc: [num_samples+1] sample boundaries
        prefill_trigger_tokens: Set of trigger token IDs for prefill phase or None
        prefill_trigger_positions: List of trigger positions for prefill phase or None
        prefill_exclude_tokens: Set of token IDs to exclude or None
        prefill_exclude_positions: List of positions to exclude or None
        device: Device to create tensors on
        
    Returns:
        mask: [total_tokens] boolean tensor indicating positions to apply intervention
    """
    total_tokens = len(sample_ids)
    
    # Get prefill sample IDs
    prefill_sample_ids = torch.nonzero(~is_decode_mask, as_tuple=False).squeeze(-1)
    
    if prefill_sample_ids.numel() == 0:
        return torch.zeros(total_tokens, dtype=torch.bool, device=device)
    
    # Check if applying to all prefill tokens (-1 means apply to all)
    if prefill_trigger_tokens is not None and -1 in prefill_trigger_tokens:
        mask = torch.isin(sample_ids, prefill_sample_ids)
        return mask
    
    # Initialize mask for prefill samples
    mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    
    # ========== 1. Position-based triggers ==========
    if prefill_trigger_positions is not None:
        # Separate positive and negative indices
        positive_positions = [p for p in prefill_trigger_positions if p >= 0]
        negative_positions = [p for p in prefill_trigger_positions if p < 0]
        
        # Compute original positions accounting for prefix caching
        if num_computed is not None:
            num_computed_expanded = num_computed[sample_ids]
            original_positions = relative_positions + num_computed_expanded
        else:
            original_positions = relative_positions
        
        # ===== Handle positive indices =====
        if positive_positions:
            trigger_positions_tensor = torch.tensor(
                positive_positions,
                dtype=original_positions.dtype,
                device=device
            )
            position_match = torch.isin(original_positions, trigger_positions_tensor)
            
            # Only apply to prefill samples
            position_match &= torch.isin(sample_ids, prefill_sample_ids)
            
            # Filter out cached positions (only process current forward)
            if num_computed is not None:
                uncached_mask = original_positions >= num_computed_expanded
                position_match &= uncached_mask
            
            mask |= position_match
        
        # ===== Handle negative indices (Python-style: -1 means last position) =====
        if negative_positions:
            # Compute each sample's total length (including cached tokens)
            sample_lengths = query_start_loc[1:] - query_start_loc[:-1]  # [num_samples]
            if num_computed is not None:
                sample_total_lengths = sample_lengths + num_computed
            else:
                sample_total_lengths = sample_lengths
            
            # For each negative index
            for neg_idx in negative_positions:
                # Convert negative index to absolute position for each sample
                # Example: length=19, neg_idx=-1 → abs_pos=18
                abs_positions_per_sample = sample_total_lengths + neg_idx  # [num_samples]
                
                # Expand to per-token: what's the target absolute position for this token's sample?
                abs_positions_expanded = abs_positions_per_sample[sample_ids]  # [total_tokens]
                
                # Match: is this token at the target absolute position?
                neg_position_match = (original_positions == abs_positions_expanded)
                
                # Only apply to prefill samples
                neg_position_match &= torch.isin(sample_ids, prefill_sample_ids)
                
                # Filter out cached positions (only process current forward)
                if num_computed is not None:
                    uncached_mask = original_positions >= num_computed_expanded
                    neg_position_match &= uncached_mask
                
                mask |= neg_position_match
    
    # ========== 2. Token-based triggers ==========
    if prefill_trigger_tokens is not None:
        trigger_tokens_tensor = torch.tensor(
            list(prefill_trigger_tokens),
            dtype=current_tokens.dtype,
            device=device
        )
        token_match = torch.isin(current_tokens, trigger_tokens_tensor)
        
        # Only apply to prefill samples
        token_match &= torch.isin(sample_ids, prefill_sample_ids)
        
        mask |= token_match
    
    # ========== 3. Apply exclusions ==========
    if mask.any():  # Only if we have positions to exclude from
        mask = apply_exclusions_gpu(
            mask=mask,
            current_tokens=current_tokens,
            sample_ids=sample_ids,
            relative_positions=relative_positions,
            num_computed=num_computed,
            prefill_sample_ids=prefill_sample_ids,
            prefill_exclude_positions=prefill_exclude_positions,
            prefill_exclude_tokens=prefill_exclude_tokens,
            query_start_loc=query_start_loc,
            device=device
        )
    
    return mask


def filter_by_generate_position(
    position_mask: torch.Tensor,
    sample_ids: torch.Tensor,
    is_decode_mask: torch.Tensor,
    num_output_tokens: torch.Tensor,
    generate_first_k: Optional[int],
    generate_after_k: Optional[int],
    device: torch.device
) -> torch.Tensor:
    """
    Filter decode positions based on generation position (first k or after k).
    
    Args:
        position_mask: [total_tokens] boolean tensor
        sample_ids: [total_tokens] sample ID for each token
        is_decode_mask: [num_samples] boolean tensor (True for decode samples)
        num_output_tokens: [num_samples] number of tokens already generated for each sample
        generate_first_k: Only intervene on first k generated tokens, or None
        generate_after_k: Start intervening from (k+1)-th token, or None
        device: Device to create tensors on
    
    Returns:
        filtered_mask: Updated boolean tensor
    """
    # Get decode sample IDs
    decode_sample_ids = torch.nonzero(is_decode_mask, as_tuple=False).squeeze(-1)
    
    if decode_sample_ids.numel() == 0:
        return position_mask
    
    # Ensure num_output_tokens is on the same device as sample_ids
    if num_output_tokens.device != device:
        num_output_tokens = num_output_tokens.to(device)
    
    # Expand num_output_tokens to per-token
    # num_output_tokens[i] is the number of tokens already generated for sample i
    # The current token being processed is the (num_output_tokens[i])-th generated token (0-indexed)
    token_gen_counts = num_output_tokens[sample_ids]  # [total_tokens]
    
    # Create decode token mask
    is_decode_token = torch.isin(sample_ids, decode_sample_ids)
    
    if generate_first_k is not None:
        # Only keep tokens where gen_count < k (0-indexed: 0, 1, ..., k-1)
        # If current token is the gen_count-th token, we apply if gen_count < k
        valid_gen_position = token_gen_counts < generate_first_k
        # Keep all prefill tokens unchanged, only filter decode tokens
        position_mask = position_mask & (~is_decode_token | valid_gen_position)
    
    elif generate_after_k is not None:
        # Only keep tokens where gen_count >= k (skip first k tokens: 0, 1, ..., k-1)
        valid_gen_position = token_gen_counts >= generate_after_k
        # Keep all prefill tokens unchanged, only filter decode tokens
        position_mask = position_mask & (~is_decode_token | valid_gen_position)
    
    return position_mask


def collect_positions_gpu_batch(
    hidden_states: torch.Tensor,
    current_tokens: torch.Tensor,
    samples_info: Dict[str, torch.Tensor],
    prefill_trigger_tokens: Optional[set],
    prefill_trigger_positions: Optional[list],
    prefill_exclude_tokens: Optional[set],
    prefill_exclude_positions: Optional[list],
    generate_trigger_tokens: Optional[set],
    generate_first_k_tokens: Optional[int],
    generate_after_k_tokens: Optional[int],
    has_prefill_triggers: bool
) -> Optional[torch.Tensor]:
    """
    Collect all intervention positions using GPU batch operations.
    
    Performs vectorized position collection with minimal GPU-CPU synchronization.
    
    Args:
        hidden_states: [total_tokens, hidden_dim]
        current_tokens: [total_tokens] token IDs
        samples_info: Dict with 'query_start_loc', 'num_computed', 'is_decode_mask'
        prefill_trigger_tokens: Set of trigger token IDs for prefill phase or None
        prefill_trigger_positions: List of trigger positions for prefill phase or None
        prefill_exclude_tokens: Set of token IDs to exclude or None
        prefill_exclude_positions: List of positions to exclude or None
        generate_trigger_tokens: Set of trigger token IDs for generation phase or None
        generate_first_k_tokens: Only intervene on first k generated tokens, or None
        generate_after_k_tokens: Start intervening from k-th token onwards, or None
        has_prefill_triggers: Whether prefill triggers are configured
        
    Returns:
        positions_tensor: [num_positions] GPU tensor of positions to transform
        or None if no positions to apply
    """
    query_start_loc = samples_info['query_start_loc']
    num_computed = samples_info['num_computed']
    is_decode_mask = samples_info['is_decode_mask']
    
    device = hidden_states.device
    total_tokens = hidden_states.shape[0]
    num_samples = len(is_decode_mask)
    
    # Normalize num_computed to torch.Tensor for consistent indexing
    if num_computed is not None and not isinstance(num_computed, torch.Tensor):
        num_computed = torch.tensor(num_computed, device=device, dtype=torch.long)
    
    # Step 1: Assign each token to its sample using binary search
    all_positions = torch.arange(total_tokens, device=device)
    # Use right=True to correctly assign boundary tokens
    # For position at query_start_loc[i], it should belong to sample i (not i-1)
    sample_ids = torch.searchsorted(query_start_loc, all_positions, right=True) - 1
    
    # Step 2: Compute relative positions within each sample
    starts = query_start_loc[:-1]
    starts_expanded = starts[sample_ids]
    relative_positions = all_positions - starts_expanded
    
    # Step 3: Initialize position mask for collecting trigger positions
    position_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    
    # Determine if decode samples should be processed
    # Either: explicit generate_trigger_tokens, or position-based control (first_k/after_k)
    has_decode_triggers = (
        generate_trigger_tokens is not None or
        generate_first_k_tokens is not None or
        generate_after_k_tokens is not None
    )
    
    # Step 4: Handle decode samples
    if has_decode_triggers and torch.any(is_decode_mask):
        # When generate_first_k_tokens or generate_after_k_tokens is set without
        # explicit generate_trigger_tokens, treat it as applying to all decode tokens
        # (the actual filtering by position happens in Step 6)
        effective_generate_trigger_tokens = generate_trigger_tokens
        if effective_generate_trigger_tokens is None and (generate_first_k_tokens is not None or generate_after_k_tokens is not None):
            effective_generate_trigger_tokens = {-1}  # -1 means all tokens
        
        decode_mask = get_decode_mask_gpu(
            current_tokens=current_tokens,
            sample_ids=sample_ids,
            is_decode_mask=is_decode_mask,
            generate_trigger_tokens=effective_generate_trigger_tokens,
            device=device
        )
        position_mask |= decode_mask
    
    # Step 5: Handle prefill samples
    if has_prefill_triggers and torch.any(~is_decode_mask):
        prefill_mask = get_prefill_mask_gpu(
            current_tokens=current_tokens,
            sample_ids=sample_ids,
            relative_positions=relative_positions,
            is_decode_mask=is_decode_mask,
            num_computed=num_computed,
            query_start_loc=query_start_loc,
            prefill_trigger_tokens=prefill_trigger_tokens,
            prefill_trigger_positions=prefill_trigger_positions,
            prefill_exclude_tokens=prefill_exclude_tokens,
            prefill_exclude_positions=prefill_exclude_positions,
            device=device
        )
        position_mask |= prefill_mask
    
    # Step 6: Apply generate position filtering (if configured)
    if generate_first_k_tokens is not None or generate_after_k_tokens is not None:
        # Extract num_output_tokens from samples_info
        num_output_tokens = samples_info.get('num_output_tokens')
        
        if num_output_tokens is not None:
            position_mask = filter_by_generate_position(
                position_mask=position_mask,
                sample_ids=sample_ids,
                is_decode_mask=is_decode_mask,
                num_output_tokens=num_output_tokens,
                generate_first_k=generate_first_k_tokens,
                generate_after_k=generate_after_k_tokens,
                device=device
            )
    
    # Step 7: Extract final positions from mask
    positions_tensor = torch.nonzero(position_mask, as_tuple=False).squeeze(-1)
    
    if positions_tensor.numel() == 0:
        return None
    
    return positions_tensor

