"""Process-local model and resource managers for the job backend."""

from .llm_manager import LLMManager
from .resource_manager import ResourceManager

llm_manager = LLMManager()
resource_manager = ResourceManager()
