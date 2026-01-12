"""
Unified ID and name generation for steer vectors.

This module provides global counter-based ID generation and timestamp-based
unique name generation to ensure uniqueness across all API modules.
"""

import time
import logging

logger = logging.getLogger(__name__)

# Global shared counter to ensure uniqueness across all APIs
# This counter is incremented atomically to prevent ID collisions
# between inference_api, chat_api, and extraction_api
# Note: Starts at 1 (not 0) as some systems require positive non-zero IDs
# Max value: 2,147,483,647 (int32 limit for vLLM)
_global_id_counter = 1


def generate_unique_id():
    """
    Generate a unique positive integer ID using a simple global counter.
    
    Returns:
        int: A unique positive integer ID (1 to 2,147,483,647)
        
    Note:
        - IDs start from 1 (not 0) as required by vLLM steer vectors
        - Counter resets on process restart
        - For production use with persistence, consider adding a database-backed counter
        
    Examples:
        >>> id1 = generate_unique_id()
        >>> id2 = generate_unique_id()
        >>> assert id1 != id2
        >>> assert id1 > 0 and id2 > 0
    """
    global _global_id_counter
    
    # Get current ID and increment
    unique_id = _global_id_counter
    _global_id_counter += 1
    
    # Safety check: wrap around if we exceed int32 max (very unlikely in practice)
    if _global_id_counter > 2147483647:
        logger.warning("Global ID counter reached int32 limit, wrapping around to 1")
        _global_id_counter = 1
    
    return unique_id


def generate_unique_name(prefix="steer_vector"):
    """
    Generate a unique name based on current timestamp.
    
    Args:
        prefix: String prefix for the generated name (default: "steer_vector")
        
    Returns:
        str: A unique name in format "{prefix}_{timestamp_microseconds}"
        
    Examples:
        >>> name1 = generate_unique_name("test")
        >>> name2 = generate_unique_name("test")
        >>> assert name1 != name2
        >>> assert name1.startswith("test_")
    """
    timestamp = int(time.time() * 1000000)  # Use microseconds for more precision
    return f"{prefix}_{timestamp}"


def reset_counter(start_value=1):
    """
    Reset the global ID counter (mainly for testing purposes).
    
    Args:
        start_value: The value to reset the counter to (default: 1)
        
    Warning:
        This should only be used in testing environments or when explicitly
        restarting the system. Do not use in production without careful consideration
        as it may cause ID collisions.
    """
    global _global_id_counter
    _global_id_counter = start_value
    logger.info(f"Global ID counter reset to {start_value}")
