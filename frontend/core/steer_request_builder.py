"""
Unified SteerVectorRequest builder.

This module provides simplified builders for creating SteerVectorRequest objects
for baseline, single-vector, and multi-vector steering configurations.
"""

import logging
from typing import List, Optional, Dict, Any
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig
from .id_generator import generate_unique_id, generate_unique_name

logger = logging.getLogger(__name__)


class SteerRequestBuilder:
    """
    Builder for SteerVectorRequest objects.
    
    This class provides static methods to simplify the creation of
    different types of steering vector requests.
    """
    
    @staticmethod
    def build_baseline_request(
        vector_path: str,
        steer_id: Optional[int] = None,
        steer_name: Optional[str] = None,
        algorithm: str = "direct",
        target_layers: Optional[List[int]] = None
    ) -> SteerVectorRequest:
        """
        Build a baseline (non-steered) request with scale=0.
        
        Args:
            vector_path: Path to a valid vector file (still required even though scale=0)
            steer_id: Optional custom ID (auto-generated if not provided)
            steer_name: Optional custom name (auto-generated if not provided)
            algorithm: Algorithm to use (must match the vector file type, default: "direct")
            target_layers: Target layers (must match the vector configuration, default: [0])
            
        Returns:
            SteerVectorRequest: A baseline request with zero steering
            
        Examples:
            >>> request = SteerRequestBuilder.build_baseline_request("/path/to/vector.safetensors")
            >>> request = SteerRequestBuilder.build_baseline_request("/path/to/loreft/", algorithm="loreft", target_layers=[8])
        """
        if steer_id is None:
            steer_id = generate_unique_id()
        if steer_name is None:
            steer_name = generate_unique_name("baseline")
        if target_layers is None:
            target_layers = [0]
        
        logger.info(f"Building baseline request: ID={steer_id}, name={steer_name}, algorithm={algorithm}, layers={target_layers}")
        
        return SteerVectorRequest(
            steer_vector_name=steer_name,
            steer_vector_int_id=steer_id,
            steer_vector_local_path=vector_path,
            scale=0.0,  # Zero scale = no steering
            target_layers=target_layers,
            algorithm=algorithm
        )
    
    @staticmethod
    def build_single_vector_request(
        vector_path: str,
        scale: float,
        target_layers: List[int],
        algorithm: str = "direct",
        steer_id: Optional[int] = None,
        steer_name: Optional[str] = None,
        prefill_trigger_tokens: Optional[List[int]] = None,
        prefill_trigger_positions: Optional[List[int]] = None,
        generate_trigger_tokens: Optional[List[int]] = None,
        normalize: bool = False,
        debug: bool = False
    ) -> SteerVectorRequest:
        """
        Build a single-vector steering request.
        
        Args:
            vector_path: Path to the vector file
            scale: Steering scale (strength)
            target_layers: List of layer indices to apply steering
            algorithm: Steering algorithm ('direct', 'loreft', etc.)
            steer_id: Optional custom ID (auto-generated if not provided)
            steer_name: Optional custom name (auto-generated if not provided)
            prefill_trigger_tokens: Token IDs that trigger steering during prefill
            prefill_trigger_positions: Position indices for prefill steering
            generate_trigger_tokens: Token IDs that trigger steering during generation
            normalize: Whether to normalize the steering vector
            debug: Whether to enable debug mode
            
        Returns:
            SteerVectorRequest: A single-vector steering request
            
        Examples:
            >>> request = SteerRequestBuilder.build_single_vector_request(
            ...     vector_path="/path/to/vector.safetensors",
            ...     scale=1.5,
            ...     target_layers=[10, 11, 12],
            ...     algorithm="direct"
            ... )
        """
        if steer_id is None:
            steer_id = generate_unique_id()
        if steer_name is None:
            steer_name = generate_unique_name("steer_vector")
        
        logger.info(f"Building single-vector request: ID={steer_id}, name={steer_name}, scale={scale}, algorithm={algorithm}")
        
        return SteerVectorRequest(
            steer_vector_name=steer_name,
            steer_vector_int_id=steer_id,
            steer_vector_local_path=vector_path,
            scale=scale,
            target_layers=target_layers,
            algorithm=algorithm,
            prefill_trigger_tokens=prefill_trigger_tokens,
            prefill_trigger_positions=prefill_trigger_positions,
            generate_trigger_tokens=generate_trigger_tokens,
            normalize=normalize,
            debug=debug
        )
    
    @staticmethod
    def build_multi_vector_request(
        vector_configs: List[Dict[str, Any]],
        conflict_resolution: str = "sequential",
        steer_id: Optional[int] = None,
        steer_name: Optional[str] = None,
        debug: bool = False
    ) -> SteerVectorRequest:
        """
        Build a multi-vector steering request.
        
        Args:
            vector_configs: List of vector configuration dictionaries, each containing:
                - path: Path to vector file (required)
                - scale: Steering scale (default: 1.0)
                - target_layers: Layer indices (optional)
                - algorithm: Algorithm (default: "direct")
                - prefill_trigger_tokens: Trigger tokens for prefill (default: [-1])
                - prefill_trigger_positions: Trigger positions (default: [-1])
                - generate_trigger_tokens: Trigger tokens for generation (default: [-1])
                - normalize: Normalize flag (default: False)
            conflict_resolution: How to resolve conflicts between vectors
                ("sequential", "average", etc.)
            steer_id: Optional custom ID (auto-generated if not provided)
            steer_name: Optional custom name (auto-generated if not provided)
            debug: Whether to enable debug mode
            
        Returns:
            SteerVectorRequest: A multi-vector steering request
            
        Examples:
            >>> configs = [
            ...     {"path": "/path/to/vector1.safetensors", "scale": 1.0, "target_layers": [10, 11]},
            ...     {"path": "/path/to/vector2.safetensors", "scale": 0.5, "target_layers": [12, 13]}
            ... ]
            >>> request = SteerRequestBuilder.build_multi_vector_request(configs)
        """
        if steer_id is None:
            steer_id = generate_unique_id()
        if steer_name is None:
            steer_name = generate_unique_name("multi_vector")
        
        logger.info(f"Building multi-vector request: ID={steer_id}, name={steer_name}, "
                   f"conflict_resolution={conflict_resolution}, num_vectors={len(vector_configs)}")
        
        # Convert dict configs to VectorConfig objects
        vector_config_objects = []
        for i, vec_config in enumerate(vector_configs):
            if 'path' not in vec_config or not vec_config['path']:
                raise ValueError(f'Vector config {i+1} is missing required "path" field')
            
            vector_config = VectorConfig(
                path=vec_config['path'],
                scale=vec_config.get('scale', 1.0),
                target_layers=vec_config.get('target_layers'),
                prefill_trigger_positions=vec_config.get('prefill_trigger_positions', [-1]),
                prefill_trigger_tokens=vec_config.get('prefill_trigger_tokens', [-1]),
                generate_trigger_tokens=vec_config.get('generate_trigger_tokens', [-1]),
                algorithm=vec_config.get('algorithm', 'direct'),
                normalize=vec_config.get('normalize', False)
            )
            vector_config_objects.append(vector_config)
            
            logger.info(f"  Vector {i+1}: path={vec_config['path']}, scale={vector_config.scale}, "
                       f"algorithm={vector_config.algorithm}, layers={vector_config.target_layers}")
        
        return SteerVectorRequest(
            steer_vector_name=steer_name,
            steer_vector_int_id=steer_id,
            vector_configs=vector_config_objects,
            conflict_resolution=conflict_resolution,
            debug=debug
        )
    
    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> SteerVectorRequest:
        """
        Build a SteerVectorRequest from a configuration dictionary.
        
        This is a convenience method that auto-detects whether to build
        a single-vector or multi-vector request based on the config structure.
        
        Args:
            config: Configuration dictionary with either:
                - Single vector: Contains 'steer_vector_local_path', 'scale', etc.
                - Multi vector: Contains 'vector_configs' list
                
        Returns:
            SteerVectorRequest: The appropriate type of request
            
        Examples:
            >>> # Single vector config
            >>> config = {
            ...     "steer_vector_local_path": "/path/to/vector.safetensors",
            ...     "scale": 1.0,
            ...     "target_layers": [10, 11, 12],
            ...     "algorithm": "direct"
            ... }
            >>> request = SteerRequestBuilder.build_from_config(config)
            
            >>> # Multi vector config
            >>> config = {
            ...     "vector_configs": [
            ...         {"path": "/path/to/v1.safetensors", "scale": 1.0},
            ...         {"path": "/path/to/v2.safetensors", "scale": 0.5}
            ...     ]
            ... }
            >>> request = SteerRequestBuilder.build_from_config(config)
        """
        # Check if this is a multi-vector config
        if 'vector_configs' in config:
            return SteerRequestBuilder.build_multi_vector_request(
                vector_configs=config['vector_configs'],
                conflict_resolution=config.get('conflict_resolution', 'sequential'),
                steer_id=config.get('steer_vector_int_id'),
                steer_name=config.get('steer_vector_name'),
                debug=config.get('debug', False)
            )
        
        # Otherwise, build single-vector request
        return SteerRequestBuilder.build_single_vector_request(
            vector_path=config['steer_vector_local_path'],
            scale=config.get('scale', 1.0),
            target_layers=config.get('target_layers', []),
            algorithm=config.get('algorithm', 'direct'),
            steer_id=config.get('steer_vector_int_id'),
            steer_name=config.get('steer_vector_name'),
            prefill_trigger_tokens=config.get('prefill_trigger_tokens'),
            prefill_trigger_positions=config.get('prefill_trigger_positions'),
            generate_trigger_tokens=config.get('generate_trigger_tokens'),
            normalize=config.get('normalize', False),
            debug=config.get('debug', False)
        )
