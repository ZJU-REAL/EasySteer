"""
Unified LLM instance management.

This module provides centralized management of vLLM model instances with
automatic caching, GPU device management, and memory cleanup capabilities.
"""

import os
import logging
from typing import Dict, Optional, Any
from vllm import LLM

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manager for LLM instances with caching and resource management.
    
    This class maintains a cache of loaded LLM instances to avoid repeated
    model loading. Instances are keyed by model path and GPU device configuration.
    
    Attributes:
        _instances: Dictionary storing cached LLM instances
    """
    
    def __init__(self):
        """Initialize the LLM manager with an empty instance cache."""
        self._instances: Dict[str, LLM] = {}
        logger.info("LLMManager initialized")
    
    def get_or_create_llm(
        self,
        model_path: str,
        gpu_devices: str = "0",
        enable_steer_vector: bool = False,
        enforce_eager: bool = True,
        enable_chunked_prefill: bool = False,
        enable_prefix_caching: bool = None,
        **kwargs
    ) -> LLM:
        """
        Get an existing LLM instance or create a new one if not cached.
        
        Args:
            model_path: Path to the model (local or HuggingFace model ID)
            gpu_devices: Comma-separated GPU device IDs (e.g., "0" or "0,1,2,3")
            enable_steer_vector: Whether to enable steering vector support
            enforce_eager: Whether to enforce eager mode (recommended for steering)
            enable_chunked_prefill: Whether to enable chunked prefill
            enable_prefix_caching: Whether to enable prefix caching (None = auto)
            **kwargs: Additional arguments to pass to LLM constructor
            
        Returns:
            LLM: The loaded or cached LLM instance
            
        Raises:
            Exception: If model loading fails
            
        Examples:
            >>> manager = LLMManager()
            >>> llm = manager.get_or_create_llm("/path/to/model", gpu_devices="0")
            >>> # Second call returns cached instance
            >>> llm2 = manager.get_or_create_llm("/path/to/model", gpu_devices="0")
            >>> assert llm is llm2
        """
        # Create a unique cache key
        key = f"{model_path}_{gpu_devices}_{enable_steer_vector}"
        
        if key in self._instances:
            logger.info(f"Returning cached LLM instance: {key}")
            return self._instances[key]
        
        try:
            # Set GPU devices environment variable
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
            logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_devices}")
            
            # Calculate tensor_parallel_size based on GPU count
            gpu_count = len(gpu_devices.split(','))
            
            # Build LLM configuration
            llm_config = {
                'model': model_path,
                'enforce_eager': enforce_eager,
                'tensor_parallel_size': gpu_count,
                'enable_chunked_prefill': enable_chunked_prefill,
            }
            
            # Add enable_steer_vector if True
            if enable_steer_vector:
                llm_config['enable_steer_vector'] = True
            
            # Add enable_prefix_caching if explicitly specified
            if enable_prefix_caching is not None:
                llm_config['enable_prefix_caching'] = enable_prefix_caching
            
            # Merge with any additional kwargs
            llm_config.update(kwargs)
            
            # Create LLM instance
            logger.info(f"Creating new LLM instance with config: {llm_config}")
            llm_instance = LLM(**llm_config)
            
            # Cache the instance
            self._instances[key] = llm_instance
            logger.info(f"Created and cached LLM instance: {key}")
            
            return llm_instance
            
        except Exception as e:
            logger.error(f"Failed to create LLM instance for {model_path}: {str(e)}")
            raise
    
    def get_instance(self, key: str) -> Optional[LLM]:
        """
        Get a specific cached LLM instance by key.
        
        Args:
            key: The cache key for the instance
            
        Returns:
            LLM instance if found, None otherwise
        """
        return self._instances.get(key)
    
    def clear_instance(self, key: str) -> bool:
        """
        Clear a specific LLM instance from cache.
        
        Args:
            key: The cache key for the instance to clear
            
        Returns:
            bool: True if instance was found and cleared, False otherwise
        """
        if key in self._instances:
            logger.info(f"Clearing LLM instance: {key}")
            try:
                del self._instances[key]
                return True
            except Exception as e:
                logger.error(f"Failed to delete LLM instance {key}: {str(e)}")
                return False
        return False
    
    def clear_all_instances(self) -> int:
        """
        Clear all cached LLM instances.
        
        Returns:
            int: Number of instances cleared
            
        Note:
            This should be called before system restart or when memory cleanup is needed.
        """
        count = len(self._instances)
        logger.info(f"Clearing all {count} LLM instances...")
        
        for key in list(self._instances.keys()):
            try:
                logger.info(f"Deleting LLM instance: {key}")
                del self._instances[key]
            except Exception as e:
                logger.error(f"Failed to delete LLM instance {key}: {str(e)}")
        
        self._instances.clear()
        logger.info(f"Cleared {count} LLM instances")
        return count
    
    def get_instance_info(self) -> Dict[str, Any]:
        """
        Get information about cached instances.
        
        Returns:
            dict: Information including count and keys of cached instances
        """
        return {
            'count': len(self._instances),
            'keys': list(self._instances.keys())
        }
    
    def __len__(self) -> int:
        """Return the number of cached instances."""
        return len(self._instances)
    
    def __contains__(self, key: str) -> bool:
        """Check if an instance with the given key is cached."""
        return key in self._instances
