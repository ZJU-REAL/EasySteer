# SPDX-License-Identifier: Apache-2.0
"""
MoE Router Logits Capture for vLLM V1

A simple, clean interface for capturing MoE router logits from vLLM V1 models.
"""

from typing import Any, Dict, Optional, List, Tuple, Union
import torch


class MoERouterLogitsCaptureV1:
    """
    MoE router logits capture for vLLM V1.
    
    This class wraps vLLM V1's RPC-based MoE router logits capture system
    to provide a simple, clean interface.
    
    Example:
        >>> import easysteer.hidden_states as hs
        >>> from vllm import LLM
        >>> 
        >>> llm = LLM(model="mistralai/Mixtral-8x7B-v0.1")
        >>> capture = hs.MoERouterLogitsCaptureV1()
        >>> router_logits, outputs = capture.get_router_logits(llm, ["Hello world"])
        >>> print(f"Captured {len(router_logits)} MoE layers")
    """
    
    def __init__(self):
        """Initialize V1 MoE capture."""
        pass
    
    def get_router_logits(
        self, 
        llm: Any, 
        texts: List[str],
        split_by_samples: bool = False
    ) -> Union[Tuple[Dict[int, torch.Tensor], Any], Tuple[List[Dict[int, torch.Tensor]], Any]]:
        """
        Get router logits from all MoE layers in vLLM V1.
        
        This method:
        1. Enables MoE router logits capture via RPC
        2. Runs inference with llm.embed()
        3. Retrieves captured router logits via RPC
        4. Cleans up (disables capture and clears memory)
        
        Args:
            llm: The vLLM LLM instance
            texts: List of input texts
            split_by_samples: Whether to split router logits by samples (default: False)
                - If True: returns List[Dict[layer_id, Tensor]] where [sample_idx][layer_id]
                - If False: returns Dict[layer_id, Tensor] (concatenated all samples)
            
        Returns:
            Tuple of (router_logits, outputs):
            - If split_by_samples=False:
                router_logits: Dict[layer_id, tensor] shape (total_tokens, n_experts)
            - If split_by_samples=True:
                router_logits: List[Dict[layer_id, tensor]] shape (seq_len, n_experts) per sample
            - outputs: vLLM inference outputs
            
        Example:
            >>> router_logits, outputs = capture.get_router_logits(
            ...     llm, ["The capital of France is"]
            ... )
            >>> 
            >>> # Access specific layer's router logits
            >>> layer_10_logits = router_logits[10]  # Shape: (num_tokens, n_experts)
            >>> 
            >>> # Analyze expert selection
            >>> probs = torch.softmax(layer_10_logits, dim=-1)
            >>> top2_probs, top2_ids = torch.topk(probs, k=2, dim=-1)
            >>> print(f"Top-2 experts: {top2_ids}")
            >>> print(f"Top-2 weights: {top2_probs}")
        """
        try:
            # Step 1: Enable MoE capture via RPC
            self._enable_capture(llm)
            
            # Step 2: Run inference
            outputs = llm.embed(texts)
            
            # Step 3: Get captured router logits via RPC
            router_logits = self._get_captured_logits(llm)
            
            # Step 4: Split by samples if requested
            if split_by_samples:
                samples_router_logits = self._split_router_logits_by_samples(
                    router_logits, outputs
                )
                return samples_router_logits, outputs
            else:
                return router_logits, outputs
                
        finally:
            # Step 5: Always cleanup
            self._cleanup(llm)
    
    def _enable_capture(self, llm: Any) -> None:
        """
        Enable MoE router logits capture via RPC.
        
        This calls the worker's enable_moe_router_logits_capture() method.
        """
        # Use V1's collective_rpc interface
        llm.llm_engine.engine_core.collective_rpc("enable_moe_router_logits_capture")
    
    def _get_captured_logits(self, llm: Any) -> Dict[int, torch.Tensor]:
        """
        Get captured router logits from workers via RPC.
        
        Returns:
            Dict mapping layer_id to router_logits tensor
        """
        # Import deserialization utility
        from vllm.hidden_states import deserialize_moe_router_logits
        
        # Call RPC to get serialized router logits
        results = llm.llm_engine.engine_core.collective_rpc("get_moe_router_logits")
        
        # Deserialize from dict format to tensors
        router_logits_dict = deserialize_moe_router_logits(results[0])
        
        return router_logits_dict
    
    def _cleanup(self, llm: Any) -> None:
        """
        Cleanup: disable capture and clear memory.
        """
        try:
            # Clear and disable via RPC
            llm.llm_engine.engine_core.collective_rpc("clear_moe_router_logits")
            llm.llm_engine.engine_core.collective_rpc("disable_moe_router_logits_capture")
        except Exception:
            # If cleanup fails, don't raise error
            pass
    
    def _split_router_logits_by_samples(
        self,
        all_router_logits: Dict[int, torch.Tensor],
        outputs: Any
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Split concatenated router logits by samples.
        
        Args:
            all_router_logits: Dict[layer_id, Tensor], each shape (total_tokens, n_experts)
            outputs: vLLM EmbeddingRequestOutput objects
            
        Returns:
            List of dicts: [sample_idx][layer_id], each shape (seq_len, n_experts)
        """
        if not all_router_logits or not outputs:
            return []
        
        # Get sample lengths from outputs
        sample_lengths = self._estimate_sample_lengths(outputs)
        
        if not sample_lengths:
            # Fallback: return all as one sample
            return [all_router_logits]
        
        # Verify total length matches
        total_length = sum(sample_lengths)
        first_layer_id = next(iter(all_router_logits.keys()))
        actual_length = all_router_logits[first_layer_id].shape[0]
        
        if total_length != actual_length:
            # Adjust last sample length to match
            diff = actual_length - total_length
            sample_lengths[-1] += diff
        
        # Split router logits by samples
        samples_router_logits = []
        start_idx = 0
        
        for sample_length in sample_lengths:
            if sample_length <= 0:
                continue
            
            end_idx = start_idx + sample_length
            
            # Extract this sample's logits for all layers
            sample_logits = {}
            for layer_id, logits_tensor in all_router_logits.items():
                sample_logits[layer_id] = logits_tensor[start_idx:end_idx]
            
            samples_router_logits.append(sample_logits)
            start_idx = end_idx
            
            if start_idx >= actual_length:
                break
        
        return samples_router_logits
    
    def _estimate_sample_lengths(self, outputs: Any) -> List[int]:
        """
        Estimate the length of each sample from outputs.
        
        For embed task, we only need prompt token counts.
        
        Args:
            outputs: vLLM EmbeddingRequestOutput objects
            
        Returns:
            List of sample lengths (in tokens)
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


def get_moe_router_logits(
    llm: Any,
    texts: List[str],
    split_by_samples: bool = False
) -> Union[Tuple[Dict[int, torch.Tensor], Any], Tuple[List[Dict[int, torch.Tensor]], Any]]:
    """
    Convenience function to get MoE router logits from vLLM V1.
    
    This is the main entry point for most users.
    
    Args:
        llm: The vLLM LLM instance (must be a MoE model)
        texts: List of input texts
        split_by_samples: Whether to split router logits by samples (default: False)
        
    Returns:
        Tuple of (router_logits, outputs)
        - If split_by_samples=False: Dict[layer_id, Tensor]
        - If split_by_samples=True: List[Dict[layer_id, Tensor]]:
        - router_logits: Dict[layer_id, tensor] where tensor is shape (num_tokens, n_experts)
        - outputs: vLLM inference outputs
        
    Example:
        >>> import easysteer.hidden_states as hs
        >>> from vllm import LLM
        >>> 
        >>> llm = LLM(model="mistralai/Mixtral-8x7B-v0.1")
        >>> router_logits, outputs = hs.get_moe_router_logits(
        ...     llm, 
        ...     ["The capital of France is Paris."]
        ... )
        >>> 
        >>> # Analyze routing for layer 10
        >>> logits = router_logits[10]
        >>> print(f"Router logits shape: {logits.shape}")  # (num_tokens, 8)
        >>> 
        >>> # Compute expert selection probabilities
        >>> probs = torch.softmax(logits, dim=-1)
        >>> print(f"Expert probabilities: {probs}")
    """
    capture = MoERouterLogitsCaptureV1()
    return capture.get_router_logits(llm, texts, split_by_samples)


