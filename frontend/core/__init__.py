"""
Core modules for EasySteer frontend backend.

This package contains shared utilities and managers used across different API modules:
- id_generator: Unified ID and name generation
- llm_manager: LLM instance management and caching
- resource_manager: Resource cleanup and backend restart functionality
- prompt_utils: Prompt formatting and tokenizer management
- steer_request_builder: SteerVectorRequest building utilities
"""

from .id_generator import generate_unique_id, generate_unique_name
from .llm_manager import LLMManager
from .resource_manager import ResourceManager
from .prompt_utils import PromptFormatter, prompt_formatter
from .steer_request_builder import SteerRequestBuilder

# Create global instances
llm_manager = LLMManager()
resource_manager = ResourceManager()

__all__ = [
    'generate_unique_id',
    'generate_unique_name',
    'LLMManager',
    'ResourceManager',
    'PromptFormatter',
    'SteerRequestBuilder',
    'llm_manager',
    'resource_manager',
    'prompt_formatter'
]
