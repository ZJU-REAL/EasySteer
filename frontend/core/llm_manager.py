"""Cache vLLM instances by effective model configuration and GPU selection."""

import hashlib
import logging
import os
import pickle
import uuid
from typing import Any, Dict, Optional

from vllm import LLM

logger = logging.getLogger(__name__)


def _ordered_config(value):
    """Stabilize containers without changing the values passed to vLLM."""
    if type(value) is dict:
        return {
            key: _ordered_config(item)
            for key, item in sorted(
                value.items(), key=lambda pair: pickle.dumps(pair[0])
            )
        }
    if type(value) is list:
        return [_ordered_config(item) for item in value]
    if type(value) is tuple:
        return tuple(_ordered_config(item) for item in value)
    return value


def _cache_key(gpu_devices: str, config: dict) -> str:
    """Snapshot all settings; unsupported custom objects must never reuse an engine.

    Pickle is only used for serialization, never loading. It captures value
    types such as torch dtypes and vLLM config objects without conflating
    them with strings. A non-serializable argument (for example a local
    callback) gets its own entry so cleanup still owns that engine.
    """
    try:
        snapshot = pickle.dumps((gpu_devices, _ordered_config(config)))
    except (
        TypeError,
        ValueError,
        pickle.PicklingError,
        AttributeError,
        RecursionError,
    ):
        logger.info(
            "LLM configuration cannot be snapshotted; creating a separate engine"
        )
        return "uncached:" + uuid.uuid4().hex
    return hashlib.sha256(snapshot).hexdigest()


class LLMManager:
    """Own cached vLLM instances keyed by all constructor arguments and GPUs."""

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
        enable_chunked_prefill: bool = None,
        enable_prefix_caching: bool = None,
        **kwargs,
    ) -> LLM:
        """Reuse a matching engine or load a new one.

        Args:
            model_path: Local model path or Hugging Face model ID.
            gpu_devices: Comma-separated GPU device IDs or UUIDs.
            enable_steer_vector: Enable steering with all supported algorithms.
            enforce_eager: Skip graph capture for faster startup of job engines.
            enable_chunked_prefill: Override chunked prefill, or use the engine
                default when None.
            enable_prefix_caching: Override prefix caching, or use the engine
                default when None.
            **kwargs: Additional LLM constructor arguments and keyword overrides.

        Returns:
            The loaded or cached LLM instance.
        """
        gpu_devices = ",".join(device.strip() for device in gpu_devices.split(","))
        if not gpu_devices or any(not device for device in gpu_devices.split(",")):
            raise ValueError("gpu_devices must contain one or more GPU IDs or UUIDs")

        # Key the effective constructor arguments, including keyword overrides.
        llm_config = {
            "model": model_path,
            "enforce_eager": enforce_eager,
            "tensor_parallel_size": len(gpu_devices.split(",")),
        }
        if enable_chunked_prefill is not None:
            llm_config["enable_chunked_prefill"] = enable_chunked_prefill
        if enable_steer_vector:
            llm_config.update(
                enable_steer_vector=True,
                steer_algorithms="all",
                steer_multi_vector=True,
            )
        if enable_prefix_caching is not None:
            llm_config["enable_prefix_caching"] = enable_prefix_caching
        llm_config.update(kwargs)
        key = _cache_key(gpu_devices, llm_config)
        if key in self._instances:
            logger.info("Returning cached LLM instance: %s", key)
            return self._instances[key]

        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
            logger.info("Set CUDA_VISIBLE_DEVICES=%s", gpu_devices)

            logger.info(f"Creating new LLM instance with config: {llm_config}")
            llm_instance = LLM(**llm_config)

            self._instances[key] = llm_instance
            logger.info(f"Created and cached LLM instance: {key}")

            return llm_instance

        except Exception as e:
            logger.error(f"Failed to create LLM instance for {model_path}: {str(e)}")
            raise

    def get_instance(self, key: str) -> Optional[LLM]:
        """Return the cached engine for a key, or None if absent."""
        return self._instances.get(key)

    def clear_instance(self, key: str) -> bool:
        """Drop a cached engine reference and report whether it existed."""
        if key in self._instances:
            logger.info(f"Clearing LLM instance: {key}")
            del self._instances[key]
            return True
        return False

    def clear_all_instances(self) -> int:
        """Drop all cached engine references and return the number removed."""
        count = len(self._instances)
        logger.info(f"Clearing all {count} LLM instances...")

        self._instances.clear()
        logger.info(f"Cleared {count} LLM instances")
        return count

    def get_instance_info(self) -> Dict[str, Any]:
        """Return the number and keys of cached engines."""
        return {"count": len(self._instances), "keys": list(self._instances.keys())}

    def __len__(self) -> int:
        """Return the number of cached instances."""
        return len(self._instances)

    def __contains__(self, key: str) -> bool:
        """Check if an instance with the given key is cached."""
        return key in self._instances
