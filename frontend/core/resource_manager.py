"""
Unified resource management and cleanup.

This module provides centralized resource cleanup, GPU memory management,
and backend restart functionality for all API modules.
"""

import os
import sys
import gc
import logging
import threading
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manager for system resources, cleanup, and restart operations.
    
    This class provides unified methods for:
    - GPU memory cleanup
    - LLM instance cleanup
    - Cache cleanup
    - Backend process restart
    """
    
    def __init__(self):
        """Initialize the resource manager."""
        logger.info("ResourceManager initialized")
    
    @staticmethod
    def cleanup_gpu_memory() -> Dict[str, Any]:
        """
        Clean up GPU memory cache.
        
        Returns:
            dict: Status information about GPU memory before and after cleanup
            
        Note:
            This uses PyTorch's CUDA cache management to free unused GPU memory.
        """
        result = {
            'success': False,
            'torch_available': False,
            'cuda_available': False,
            'gpu_info': []
        }
        
        try:
            import torch
            result['torch_available'] = True
            
            if torch.cuda.is_available():
                result['cuda_available'] = True
                logger.info("Clearing CUDA cache...")
                
                # Get memory info before cleanup
                for i in range(torch.cuda.device_count()):
                    before_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    before_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    
                    # Clear cache
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Get memory info after cleanup
                    after_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    after_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    
                    gpu_info = {
                        'gpu_id': i,
                        'before': {
                            'allocated_gb': round(before_allocated, 2),
                            'reserved_gb': round(before_reserved, 2)
                        },
                        'after': {
                            'allocated_gb': round(after_allocated, 2),
                            'reserved_gb': round(after_reserved, 2)
                        },
                        'freed_gb': round(before_reserved - after_reserved, 2)
                    }
                    result['gpu_info'].append(gpu_info)
                    logger.info(f"GPU {i} - Before: {before_allocated:.2f}GB allocated, {before_reserved:.2f}GB reserved")
                    logger.info(f"GPU {i} - After: {after_allocated:.2f}GB allocated, {after_reserved:.2f}GB reserved")
                
                result['success'] = True
            else:
                logger.info("CUDA not available, skipping GPU cleanup")
                
        except ImportError:
            logger.info("PyTorch not available, skipping CUDA cache cleanup")
        except Exception as e:
            logger.error(f"Error clearing CUDA cache: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def cleanup_llm_instances() -> Dict[str, int]:
        """
        Clean up all LLM instances cached by the shared core llm_manager.

        Returns:
            dict: Count of instances cleaned up
        """
        result = {'total_count': 0}
        try:
            from core import llm_manager
            result['total_count'] = llm_manager.clear_all_instances()
        except Exception as e:
            logger.error(f"Error cleaning up LLM instances: {str(e)}")
        return result

    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """
        Force Python garbage collection.
        
        Returns:
            dict: Information about objects collected
        """
        logger.info("Running garbage collection...")
        collected = gc.collect()
        result = {
            'objects_collected': collected,
            'garbage_count': len(gc.garbage)
        }
        logger.info(f"Garbage collection completed: {collected} objects collected")
        return result
    
    @staticmethod
    def cleanup_all_resources() -> Dict[str, Any]:
        """
        Perform complete resource cleanup.
        
        This method:
        1. Cleans up LLM instances
        2. Forces garbage collection
        3. Cleans up GPU memory

        Returns:
            dict: Detailed information about cleanup operations
        """
        logger.info("Starting complete resource cleanup...")

        result = {
            'timestamp': time.time(),
            'llm_cleanup': {},
            'gc_info': {},
            'gpu_cleanup': {}
        }

        # Step 1: Clean up LLM instances
        result['llm_cleanup'] = ResourceManager.cleanup_llm_instances()

        # Step 2: Force garbage collection
        result['gc_info'] = ResourceManager.force_garbage_collection()

        # Step 3: Clean up GPU memory
        result['gpu_cleanup'] = ResourceManager.cleanup_gpu_memory()
        
        logger.info("Complete resource cleanup finished")
        return result
    
    @staticmethod
    def restart_backend(delay: float = 1.0) -> Dict[str, Any]:
        """
        Restart the backend process with proper resource cleanup.
        
        Args:
            delay: Delay in seconds before restart (to allow response to be sent)
            
        Returns:
            dict: Status information about the restart operation
            
        Note:
            This creates a background thread to perform cleanup and restart,
            allowing the response to be sent to the client first.
        """
        logger.info("Preparing to fully restart the backend process...")
        
        def delayed_restart():
            """Delayed restart to allow response to be sent"""
            time.sleep(delay)
            logger.info("Starting backend restart sequence...")
            
            # Perform complete resource cleanup
            cleanup_result = ResourceManager.cleanup_all_resources()
            logger.info(f"Resource cleanup completed: {cleanup_result}")
            
            # Get current Python executable and script arguments
            python_executable = sys.executable
            script_args = sys.argv
            
            # Use os.execv to restart the process
            logger.info("Executing process restart...")
            logger.info(f"Command: {python_executable} {' '.join(script_args)}")
            os.execv(python_executable, [python_executable] + script_args)
        
        # Execute restart in a new thread to avoid blocking the response
        restart_thread = threading.Thread(target=delayed_restart)
        restart_thread.daemon = True
        restart_thread.start()
        
        return {
            "success": True,
            "message": "Backend is restarting and cleaning up GPU memory, please try again in a few seconds...",
            "delay_seconds": delay
        }
