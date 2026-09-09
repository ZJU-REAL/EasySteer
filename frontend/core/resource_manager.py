"""Release job resources and restart the backend process."""

import gc
import logging
import os
import sys
import threading
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manage cached engines, GPU memory, and backend restarts."""

    def __init__(self):
        """Initialize the resource manager."""
        logger.info("ResourceManager initialized")

    @staticmethod
    def cleanup_gpu_memory() -> Dict[str, Any]:
        """Release unused PyTorch CUDA cache memory.

        Returns:
            Availability, cleanup status, and GPU memory usage before and after.
        """
        result = {
            "success": False,
            "torch_available": False,
            "cuda_available": False,
            "gpu_info": [],
        }

        try:
            import torch

            result["torch_available"] = True

            if torch.cuda.is_available():
                result["cuda_available"] = True
                logger.info("Clearing CUDA cache...")

                for i in range(torch.cuda.device_count()):
                    before_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    before_reserved = torch.cuda.memory_reserved(i) / 1024**3

                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    after_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    after_reserved = torch.cuda.memory_reserved(i) / 1024**3

                    gpu_info = {
                        "gpu_id": i,
                        "before": {
                            "allocated_gb": round(before_allocated, 2),
                            "reserved_gb": round(before_reserved, 2),
                        },
                        "after": {
                            "allocated_gb": round(after_allocated, 2),
                            "reserved_gb": round(after_reserved, 2),
                        },
                        "freed_gb": round(before_reserved - after_reserved, 2),
                    }
                    result["gpu_info"].append(gpu_info)
                    logger.info(
                        f"GPU {i} - Before: {before_allocated:.2f}GB allocated, {before_reserved:.2f}GB reserved"
                    )
                    logger.info(
                        f"GPU {i} - After: {after_allocated:.2f}GB allocated, {after_reserved:.2f}GB reserved"
                    )

                result["success"] = True
            else:
                logger.info("CUDA not available, skipping GPU cleanup")

        except ImportError:
            logger.info("PyTorch not available, skipping CUDA cache cleanup")
        except Exception as e:
            logger.error(f"Error clearing CUDA cache: {str(e)}")
            result["error"] = str(e)

        return result

    @staticmethod
    def cleanup_llm_instances() -> Dict[str, int]:
        """Release engines cached by the shared LLM manager and report the count."""
        result = {"total_count": 0}
        try:
            from .runtime import llm_manager

            result["total_count"] = llm_manager.clear_all_instances()
        except Exception as e:
            logger.error(f"Error cleaning up LLM instances: {str(e)}")
        return result

    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """Collect unreachable Python objects and report collection counts."""
        logger.info("Running garbage collection...")
        collected = gc.collect()
        result = {"objects_collected": collected, "garbage_count": len(gc.garbage)}
        logger.info(f"Garbage collection completed: {collected} objects collected")
        return result

    @staticmethod
    def cleanup_all_resources() -> Dict[str, Any]:
        """Release cached engines, then collect objects and clear CUDA caches.

        Returns:
            The timestamp and results of each cleanup operation.
        """
        logger.info("Starting complete resource cleanup...")

        result = {
            "timestamp": time.time(),
            "llm_cleanup": {},
            "gc_info": {},
            "gpu_cleanup": {},
        }

        result["llm_cleanup"] = ResourceManager.cleanup_llm_instances()

        result["gc_info"] = ResourceManager.force_garbage_collection()

        result["gpu_cleanup"] = ResourceManager.cleanup_gpu_memory()

        logger.info("Complete resource cleanup finished")
        return result

    @staticmethod
    def restart_backend(delay: float = 1.0) -> Dict[str, Any]:
        """Clean up and restart in a background thread after sending the response.

        Args:
            delay: Seconds to wait before cleanup and restart.

        Returns:
            Restart acknowledgement and the configured delay.
        """
        logger.info("Preparing to fully restart the backend process...")

        def delayed_restart():
            """Delayed restart to allow response to be sent"""
            time.sleep(delay)
            logger.info("Starting backend restart sequence...")

            cleanup_result = ResourceManager.cleanup_all_resources()
            logger.info(f"Resource cleanup completed: {cleanup_result}")

            if os.environ.get("SERVER_SOFTWARE", "").startswith("gunicorn/"):
                # The Gunicorn master owns the listening sockets and starts
                # a fresh worker when this one exits. Re-executing its CLI
                # here would instead try to launch a second master.
                logger.info("Exiting job worker for Gunicorn to restart it...")
                os._exit(0)

            python_executable = sys.executable
            script_args = sys.argv

            logger.info("Executing process restart...")
            logger.info(f"Command: {python_executable} {' '.join(script_args)}")
            os.execv(python_executable, [python_executable] + script_args)

        restart_thread = threading.Thread(target=delayed_restart)
        restart_thread.daemon = True
        restart_thread.start()

        return {
            "success": True,
            "message": "Backend is restarting and cleaning up GPU memory, please try again in a few seconds...",
            "delay_seconds": delay,
        }
