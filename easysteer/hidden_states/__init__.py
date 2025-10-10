# SPDX-License-Identifier: Apache-2.0
"""
Transformer Hidden States Capture Library

A library for capturing hidden states from transformer models using vLLM.
"""

from .core.capture import HiddenStatesCapture
from .core.storage import HiddenStatesStore, HiddenStatesCaptureContext
from .core.wrapper import TransformerLayerWrapper, wrap_transformer_layers
from .detection.patterns import get_layer_patterns_for_model
from .detection.structure import detect_layer_structure, get_optimal_layer_pattern
from .adapters.base import LLMAdapter
from .adapters.vllm import VLLMAdapter

# Convenience functions
def get_all_hidden_states(llm, texts, adapter=None, split_by_samples=True):
    """
    Convenience function to get all hidden states from vLLM
    
    Args:
        llm: The vLLM LLM instance
        texts: List of input texts
        adapter: LLMAdapter instance (uses VLLMAdapter if None)
        split_by_samples: Whether to split hidden states by samples
        
    Returns:
        Tuple of (hidden_states, outputs)
    """
    if adapter is None:
        adapter = _auto_detect_adapter(llm)
    
    # Use global store to ensure consistency across multiple calls
    from .core.storage import get_global_store
    store = get_global_store()
    
    capture = HiddenStatesCapture(adapter=adapter, store=store)
    return capture.get_all_hidden_states(llm, texts, split_by_samples=split_by_samples)

def _auto_detect_adapter(llm):
    """Auto-detect the appropriate adapter for the LLM"""
    llm_type = type(llm).__name__
    module_name = type(llm).__module__
    
    if 'vllm' in module_name.lower() or hasattr(llm, 'llm_engine'):
        return VLLMAdapter()
    else:
        raise ValueError(f"Only vLLM is supported. LLM type: {llm_type}")

__all__ = [
    # Core classes
    "HiddenStatesCapture",
    "HiddenStatesStore", 
    "HiddenStatesCaptureContext",
    "TransformerLayerWrapper",
    "wrap_transformer_layers",
    
    # Detection functions
    "get_layer_patterns_for_model",
    "detect_layer_structure",
    "get_optimal_layer_pattern",
    
    # Adapters
    "LLMAdapter",
    "VLLMAdapter", 
    
    # Convenience functions
    "get_all_hidden_states",
]

__version__ = "1.0.0" 