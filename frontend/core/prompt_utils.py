"""
Unified prompt formatting utilities.

This module provides centralized prompt generation for different model types
with automatic tokenizer management and caching.
"""

import logging
from typing import List, Dict, Optional
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class PromptFormatter:
    """
    Manager for prompt formatting with tokenizer caching.
    
    This class handles:
    - Model-specific prompt formatting (Gemma, Qwen, Llama, Mistral, etc.)
    - Single-turn and multi-turn conversation formatting
    - Tokenizer caching and management
    """
    
    def __init__(self):
        """Initialize the prompt formatter with an empty tokenizer cache."""
        self._tokenizer_cache: Dict[str, Optional[AutoTokenizer]] = {}
        logger.info("PromptFormatter initialized")
    
    def get_tokenizer(self, model_path: str) -> Optional[AutoTokenizer]:
        """
        Get or load a tokenizer for the specified model.
        
        Args:
            model_path: Path to the model (local or HuggingFace model ID)
            
        Returns:
            AutoTokenizer instance if successful, None if loading fails
        """
        if model_path not in self._tokenizer_cache:
            try:
                self._tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"Loaded tokenizer for model: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for {model_path}, using fallback templates: {str(e)}")
                self._tokenizer_cache[model_path] = None
        
        return self._tokenizer_cache[model_path]
    
    def clear_tokenizer_cache(self) -> int:
        """
        Clear all cached tokenizers.
        
        Returns:
            int: Number of tokenizers cleared
        """
        count = len(self._tokenizer_cache)
        self._tokenizer_cache.clear()
        logger.info(f"Cleared {count} tokenizers from cache")
        return count
    
    def format_single_turn(self, model_path: str, instruction: str) -> str:
        """
        Format a single-turn prompt based on model type.
        
        Args:
            model_path: Path to the model
            instruction: The user instruction/query
            
        Returns:
            str: Formatted prompt ready for generation
            
        Examples:
            >>> formatter = PromptFormatter()
            >>> prompt = formatter.format_single_turn("/path/to/gemma", "Hello!")
        """
        model_path_lower = model_path.lower()
        tokenizer = self.get_tokenizer(model_path)
        
        # For Gemma models, use apply_chat_template if available
        if 'gemma' in model_path_lower:
            if tokenizer:
                messages = [{"role": "user", "content": instruction}]
                try:
                    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception as e:
                    logger.warning(f"Failed to apply Gemma chat template: {str(e)}")
            # Fallback for Gemma models
            return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        
        # For Qwen models
        elif 'qwen' in model_path_lower:
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        # For Llama models
        elif 'llama' in model_path_lower:
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        # For Mistral models
        elif 'mistral' in model_path_lower:
            return f"[INST] {instruction} [/INST]"
        
        # Default fallback (try tokenizer's chat template if available)
        else:
            if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": instruction}]
                try:
                    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except:
                    pass
            # Last resort fallback
            return f"User: {instruction}\nAssistant:"
    
    def format_multi_turn(
        self,
        model_path: str,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format a multi-turn conversation prompt based on model type.
        
        Args:
            model_path: Path to the model
            message: The current user message
            history: List of previous conversation turns with 'role' and 'content' keys
            
        Returns:
            str: Formatted prompt including conversation history
            
        Examples:
            >>> formatter = PromptFormatter()
            >>> history = [
            ...     {"role": "user", "content": "Hi"},
            ...     {"role": "assistant", "content": "Hello!"}
            ... ]
            >>> prompt = formatter.format_multi_turn("/path/to/model", "How are you?", history)
        """
        model_path_lower = model_path.lower()
        prompt = ""
        
        # Process conversation history
        if history and len(history) > 0:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                
                if 'gemma' in model_path_lower:
                    if role == "user":
                        prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                    else:
                        prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
                
                elif 'qwen' in model_path_lower:
                    if role == "user":
                        prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                    else:
                        prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
                else:
                    # Generic format
                    if role == "user":
                        prompt += f"User: {content}\n"
                    else:
                        prompt += f"Assistant: {content}\n"
        
        # Add current message
        if 'gemma' in model_path_lower:
            prompt += f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model"
        elif 'qwen' in model_path_lower:
            prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant"
        else:
            prompt += f"User: {message}\nAssistant:"
        
        return prompt
    
    def __len__(self) -> int:
        """Return the number of cached tokenizers."""
        return len(self._tokenizer_cache)


# Global instance for easy import
prompt_formatter = PromptFormatter()
