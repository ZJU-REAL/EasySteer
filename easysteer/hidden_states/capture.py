# SPDX-License-Identifier: Apache-2.0
"""
Core hidden states capture logic for vLLM V1

This module encapsulates all RPC calls and provides a clean interface
for capturing hidden states from vLLM V1 models.
"""

from typing import Any, List, Tuple, Union
import torch


class HiddenStatesCaptureV1:
    """
    Hidden states capture for vLLM V1.
    
    This class wraps vLLM V1's RPC-based hidden states capture system
    to provide a simple, clean interface.
    
    Example:
        >>> capture = HiddenStatesCaptureV1()
        >>> hidden_states, outputs = capture.get_all_hidden_states(llm, texts)
    """
    
    def __init__(self):
        """Initialize V1 capture."""
        pass
    
    def get_all_hidden_states(
        self, 
        llm: Any, 
        texts: List[str],
        split_by_samples: bool = True
    ) -> Union[Tuple[List[List[torch.Tensor]], Any], Tuple[List[torch.Tensor], Any]]:
        """
        Get all hidden states from vLLM V1.
        
        This method:
        1. Enables hidden states capture via RPC
        2. Runs inference with llm.embed()
        3. Retrieves captured hidden states via RPC
        4. Optionally splits by samples
        5. Cleans up (disables capture and clears memory)
        
        Args:
            llm: The vLLM LLM instance
            texts: List of input texts
            split_by_samples: Whether to split hidden states by samples
                - If True: returns List[List[Tensor]] where [sample_idx][layer_idx]
                - If False: returns List[Tensor] where [layer_idx] (concatenated)
            
        Returns:
            Tuple of (hidden_states, outputs):
            - If split_by_samples=True:
                hidden_states[sample_idx][layer_idx] is shape (seq_len, hidden_size)
            - If split_by_samples=False:
                hidden_states[layer_idx] is shape (total_tokens, hidden_size)
            - outputs: vLLM inference outputs
            
        Example:
            >>> # Get hidden states split by samples
            >>> hidden_states, outputs = capture.get_all_hidden_states(
            ...     llm, ["Hello", "World"], split_by_samples=True
            ... )
            >>> print(f"Sample 0, Layer 0: {hidden_states[0][0].shape}")
            >>> 
            >>> # Get concatenated hidden states
            >>> hidden_states, outputs = capture.get_all_hidden_states(
            ...     llm, ["Hello", "World"], split_by_samples=False
            ... )
            >>> print(f"Layer 0 (all samples): {hidden_states[0].shape}")
        """
        try:
            # Step 1: Enable capture via RPC
            self._enable_capture(llm)
            
            # Step 2: Run inference
            outputs = llm.embed(texts)
            
            # Step 3: Get captured hidden states via RPC
            all_hidden_states = self._get_captured_states(llm)
            
            # Step 4: Split by samples if requested
            if split_by_samples:
                samples_hidden_states = self._split_hidden_states_by_samples(
                    all_hidden_states, outputs
                )
                return samples_hidden_states, outputs
            else:
                return all_hidden_states, outputs
                
        finally:
            # Step 5: Always cleanup
            self._cleanup(llm)
    
    def _enable_capture(self, llm: Any) -> None:
        """
        Enable hidden states capture via RPC.
        
        This calls the worker's enable_hidden_states_capture() method,
        which enables capture, clears previous states, and enables multi-batch mode.
        """
        llm.llm_engine.engine_core.collective_rpc("enable_hidden_states_capture")
    
    def _get_captured_states(self, llm: Any) -> List[torch.Tensor]:
        """
        Get captured hidden states from workers via RPC.
        
        This method:
        1. Calls RPC to get serialized hidden states
        2. Deserializes from dict format to tensors
        3. Converts to list format ordered by layer index
        
        Returns:
            List of tensors, one per layer, ordered by layer index.
            Each tensor has shape (total_tokens, hidden_size).
        """
        # Import deserialization utility from vLLM
        from vllm.hidden_states import deserialize_hidden_states
        
        # Call RPC to get serialized hidden states
        results = llm.llm_engine.engine_core.collective_rpc("get_captured_hidden_states")
        
        # Deserialize to dict[layer_id, tensor]
        hidden_states_dict = deserialize_hidden_states(results[0])
        
        # Convert dict to list ordered by layer_id
        sorted_layer_ids = sorted(hidden_states_dict.keys())
        hidden_states_list = [hidden_states_dict[layer_id] for layer_id in sorted_layer_ids]
        
        return hidden_states_list
    
    def _cleanup(self, llm: Any) -> None:
        """
        Cleanup after capture - disable and clear hidden states.
        
        This frees up memory by clearing captured states and disabling capture.
        Errors during cleanup are silently ignored.
        """
        try:
            llm.llm_engine.engine_core.collective_rpc("clear_hidden_states")
            llm.llm_engine.engine_core.collective_rpc("disable_hidden_states_capture")
        except Exception:
            # Ignore cleanup errors
            pass
    
    def _split_hidden_states_by_samples(
        self, 
        all_hidden_states: List[torch.Tensor],
        outputs: Any
    ) -> List[List[torch.Tensor]]:
        """
        Split concatenated hidden states by samples.
        
        Args:
            all_hidden_states: List of tensors [layer_idx], each shape (total_tokens, hidden_size)
            outputs: vLLM inference outputs
            
        Returns:
            List of lists: [sample_idx][layer_idx], each shape (seq_len, hidden_size)
        """
        if not all_hidden_states or not outputs:
            return []
        
        # Get sample lengths from outputs
        sample_lengths = self._estimate_sample_lengths(outputs)
        
        if not sample_lengths:
            # Fallback: return all as one sample
            return [all_hidden_states]
        
        # Verify total length matches
        total_length = sum(sample_lengths)
        actual_length = all_hidden_states[0].shape[0] if all_hidden_states else 0
        
        if total_length != actual_length:
            # Adjust last sample length to match
            diff = actual_length - total_length
            sample_lengths[-1] += diff
        
        # Split hidden states by samples
        samples_hidden_states = []
        start_idx = 0
        
        for sample_length in sample_lengths:
            if sample_length <= 0:
                continue
                
            end_idx = start_idx + sample_length
            
            # Collect all layers for this sample
            sample_all_layers = []
            for layer_hidden_states in all_hidden_states:
                sample_layer_hidden_states = layer_hidden_states[start_idx:end_idx]
                sample_all_layers.append(sample_layer_hidden_states)
            
            samples_hidden_states.append(sample_all_layers)
            start_idx = end_idx
            
            if start_idx >= actual_length:
                break
        
        return samples_hidden_states
    
    def _estimate_sample_lengths(self, outputs: Any) -> List[int]:
        """
        Estimate the length of each sample from outputs.
        
        Args:
            outputs: vLLM inference outputs
            
        Returns:
            List of sample lengths
        """
        if not outputs:
            return []
        
        sample_lengths = []
        for output in outputs:
            if hasattr(output, 'prompt_token_ids'):
                # EmbeddingRequestOutput format
                sample_lengths.append(len(output.prompt_token_ids))
            elif hasattr(output, 'token_ids'):
                # Other output formats
                sample_lengths.append(len(output.token_ids))
            else:
                # Fallback: assume 1 token
                sample_lengths.append(1)
        
        return sample_lengths


# Convenience function (top-level API)
def get_all_hidden_states(
    llm: Any, 
    texts: List[str],
    split_by_samples: bool = True
) -> Union[Tuple[List[List[torch.Tensor]], Any], Tuple[List[torch.Tensor], Any]]:
    """
    Convenience function to get all hidden states from vLLM V1.
    
    This is the main entry point for users. It automatically handles
    enabling capture, running inference, retrieving states, and cleanup.
    
    Args:
        llm: The vLLM LLM instance
        texts: List of input texts to process
        split_by_samples: Whether to split hidden states by samples
            - If True: returns List[List[Tensor]] where [sample_idx][layer_idx]
            - If False: returns List[Tensor] where [layer_idx] (concatenated)
        
    Returns:
        Tuple of (hidden_states, outputs):
        - If split_by_samples=True:
            hidden_states[sample_idx][layer_idx] is shape (seq_len, hidden_size)
        - If split_by_samples=False:
            hidden_states[layer_idx] is shape (total_tokens, hidden_size)
        - outputs: vLLM inference outputs
        
    Example:
        >>> import easysteer.hidden_states_v1 as hs
        >>> from vllm import LLM
        >>> import os
        >>> 
        >>> os.environ["VLLM_USE_V1"] = "1"
        >>> llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
        >>> 
        >>> # Get hidden states split by samples
        >>> all_hidden_states, outputs = hs.get_all_hidden_states(
        ...     llm, ["Hello world", "How are you?"]
        ... )
        >>> 
        >>> print(f"Captured {len(all_hidden_states)} samples")
        >>> print(f"Sample 0 has {len(all_hidden_states[0])} layers")
        >>> print(f"Sample 0, Layer 0 shape: {all_hidden_states[0][0].shape}")
        >>> 
        >>> # Get concatenated hidden states (all samples together)
        >>> all_layers, outputs = hs.get_all_hidden_states(
        ...     llm, ["Hello world", "How are you?"], split_by_samples=False
        ... )
        >>> 
        >>> print(f"Total layers: {len(all_layers)}")
        >>> print(f"Layer 0 shape (concatenated): {all_layers[0].shape}")
    """
    capture = HiddenStatesCaptureV1()
    return capture.get_all_hidden_states(llm, texts, split_by_samples)

