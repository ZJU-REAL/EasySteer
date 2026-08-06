"""
Core modules for the EasySteer job backend (extraction / training / SAE).

This package contains shared utilities and managers used across the API
modules:
- llm_manager: LLM instance management and caching
- resource_manager: Resource cleanup and backend restart functionality
- messages: Bilingual messages and Accept-Language parsing
- validation: Request payload validation
- config_store: JSON config preset listing/loading
- local_imports: Temporary sys.path handling for repo-local imports

Text generation and steering are served by the vllm-steer
OpenAI-compatible server; this backend only runs long-lived jobs.
"""

from .llm_manager import LLMManager
from .resource_manager import ResourceManager
from .messages import get_message, lang
from .validation import require_fields
from .config_store import ConfigStore
from .local_imports import PROJECT_ROOT, project_root_on_path

# Create global instances
llm_manager = LLMManager()
resource_manager = ResourceManager()

__all__ = [
    'LLMManager',
    'ResourceManager',
    'get_message',
    'lang',
    'require_fields',
    'ConfigStore',
    'PROJECT_ROOT',
    'project_root_on_path',
    'llm_manager',
    'resource_manager',
]
