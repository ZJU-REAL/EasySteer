"""
Core modules for EasySteer frontend backend.

This package contains shared utilities and managers used across different API modules:
- id_generator: Unified ID and name generation
- llm_manager: LLM instance management and caching
- resource_manager: Resource cleanup and backend restart functionality
- prompt_utils: Prompt formatting and tokenizer management
- steer_request_builder: SteeringSpec building from legacy request fields
- messages: Bilingual messages and Accept-Language parsing
- validation: Request payload validation
- config_store: JSON config preset listing/loading
- local_imports: Temporary sys.path handling for repo-local imports
"""

from .id_generator import generate_unique_id, generate_unique_name
from .llm_manager import LLMManager
from .resource_manager import ResourceManager
from .prompt_utils import PromptFormatter, prompt_formatter
from .steer_request_builder import (
    build_apply_specs,
    build_multi_vector_spec,
    build_single_vector_spec,
)
from .messages import get_message, lang
from .validation import require_fields
from .config_store import ConfigStore
from .local_imports import PROJECT_ROOT, project_root_on_path

# Create global instances
llm_manager = LLMManager()
resource_manager = ResourceManager()

__all__ = [
    'generate_unique_id',
    'generate_unique_name',
    'LLMManager',
    'ResourceManager',
    'PromptFormatter',
    'build_apply_specs',
    'build_multi_vector_spec',
    'build_single_vector_spec',
    'get_message',
    'lang',
    'require_fields',
    'ConfigStore',
    'PROJECT_ROOT',
    'project_root_on_path',
    'llm_manager',
    'resource_manager',
    'prompt_formatter',
]
