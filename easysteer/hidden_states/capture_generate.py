# SPDX-License-Identifier: Apache-2.0
"""
Hidden States Capture for Generate Task

Extends hidden states capture to work with generate task, not just embed.
This is useful for models that don't support embed task (e.g., multimodal models).
"""

from typing import Any, Dict, Optional, List, Union, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


class HiddenStatesCaptureGenerate:
    """
    Hidden states capture for generate task.
    
    This works with llm.generate() instead of llm.embed().
    Useful for models that don't support embed task (e.g., Qwen-VL, LLaVA, etc.)
    """
    
    def __init__(self):
        """Initialize generate-mode hidden states capture."""
        pass
    
    def get_all_hidden_states_generate(
        self, 
        llm: Any, 
        prompts: Union[List[str], List[Dict[str, Any]]],
        max_tokens: int = 1,
        split_by_samples: bool = True,
        **generate_kwargs
    ) -> Union[Tuple[List[List[torch.Tensor]], Any], Tuple[List[torch.Tensor], Any]]:
        """
        Get all hidden states from all layers using generate task.
        
        适用于不支持embed任务的模型（如Qwen-VL、LLaVA等多模态模型）。
        通过设置max_tokens=1来实现和embed模型相同的效果。
        
        Args:
            llm: The vLLM LLM instance (must NOT specify task="embed")
            prompts: List of input prompts,支持以下格式:
                - List[str]: 纯文本输入
                - List[Dict]: 多模态输入，例如包含text和image
            max_tokens: Number of tokens to generate (default 1, just to trigger forward)
            split_by_samples: Whether to split hidden states by samples (default: True)
                - If True: returns List[List[Tensor]] where [sample_idx][layer_idx]
                - If False: returns List[Tensor] where [layer_idx] (concatenated)
            **generate_kwargs: Additional arguments passed to llm.generate()
                (e.g., temperature, top_p, etc.)
            
        Returns:
            Tuple of (hidden_states, outputs):
            - If split_by_samples=True:
                hidden_states[sample_idx][layer_idx] is shape (seq_len, hidden_size)
            - If split_by_samples=False:
                hidden_states[layer_idx] is shape (total_tokens, hidden_size)
            - outputs: vLLM generation outputs
            
        Examples:
            >>> from vllm import LLM
            >>> import easysteer.hidden_states as hs
            >>> 
            >>> # 文本输入
            >>> llm = LLM(model="Qwen2.5-VL-7B-Instruct", tensor_parallel_size=1)
            >>> hidden_states, outputs = hs.get_all_hidden_states_generate(
            ...     llm, 
            ...     prompts=["What is AI?"],
            ...     max_tokens=1
            ... )
            >>> 
            >>> # 多模态输入 (text + image)
            >>> multimodal_prompts = [
            ...     {
            ...         "prompt": "Describe this image",
            ...         "multi_modal_data": {"image": image_url}
            ...     }
            ... ]
            >>> hidden_states, outputs = hs.get_all_hidden_states_generate(
            ...     llm,
            ...     prompts=multimodal_prompts,
            ...     max_tokens=1,
            ...     split_by_samples=True
            ... )
        """
        try:
            # Step 1: Enable hidden states capture via RPC
            self._enable_capture(llm)
            
            # Step 2: Run generation (只生成max_tokens个token来触发forward)
            from vllm import SamplingParams
            
            # 准备sampling params
            sampling_params_kwargs = {
                'max_tokens': max_tokens,
                'temperature': generate_kwargs.get('temperature', 0.0),
            }
            # 添加其他generate_kwargs (排除已处理的)
            for key, value in generate_kwargs.items():
                if key not in ['max_tokens', 'temperature']:
                    sampling_params_kwargs[key] = value
            
            sampling_params = SamplingParams(**sampling_params_kwargs)
            
            # 生成 - prompts可以是List[str]或List[Dict]（多模态）
            outputs = llm.generate(prompts, sampling_params)
            
            # Step 3: Get captured hidden states via RPC
            all_hidden_states = self._get_captured_states(llm)
            
            # Step 4: Split by samples if requested
            if split_by_samples:
                samples_hidden_states = self._split_hidden_states_by_samples(
                    all_hidden_states, outputs
                )
                return samples_hidden_states, outputs
            else:
                return all_hidden_states, outputs
                
        finally:
            # Step 5: Always cleanup
            self._cleanup(llm)
    
    def _enable_capture(self, llm: Any) -> None:
        """Enable hidden states capture via RPC."""
        llm.llm_engine.engine_core.collective_rpc("enable_hidden_states_capture")
    
    def _get_captured_states(self, llm: Any) -> List[torch.Tensor]:
        """
        Get captured hidden states from workers via RPC.
        
        Returns:
            List of tensors, one per layer, ordered by layer index.
            Each tensor has shape (total_tokens, hidden_size).
        """
        # Import deserialization utility from vLLM
        from vllm.hidden_states import deserialize_hidden_states
        
        # Call RPC to get serialized hidden states
        results = llm.llm_engine.engine_core.collective_rpc("get_captured_hidden_states")
        
        # Deserialize to dict[layer_id, tensor]
        hidden_states_dict = deserialize_hidden_states(results[0])
        
        # Convert dict to list ordered by layer_id
        sorted_layer_ids = sorted(hidden_states_dict.keys())
        hidden_states_list = [hidden_states_dict[layer_id] for layer_id in sorted_layer_ids]
        
        return hidden_states_list
    
    def _cleanup(self, llm: Any) -> None:
        """Cleanup: disable capture and clear memory."""
        try:
            llm.llm_engine.engine_core.collective_rpc("clear_hidden_states")
            llm.llm_engine.engine_core.collective_rpc("disable_hidden_states_capture")
        except Exception:
            pass
    
    def _split_hidden_states_by_samples(
        self,
        all_hidden_states: List[torch.Tensor],
        outputs: Any
    ) -> List[List[torch.Tensor]]:
        """
        Split concatenated hidden states by samples.
        
        Args:
            all_hidden_states: List of tensors [layer_idx], each shape (total_tokens, hidden_size)
            outputs: vLLM RequestOutput objects
            
        Returns:
            List of lists: [sample_idx][layer_idx], each shape (seq_len, hidden_size)
        """
        if not all_hidden_states or not outputs:
            return []
        
        # Get sample lengths from outputs
        sample_lengths = self._estimate_sample_lengths(outputs)
        
        if not sample_lengths:
            # Fallback: return all as one sample
            return [all_hidden_states]
        
        # Verify total length matches
        total_length = sum(sample_lengths)
        actual_length = all_hidden_states[0].shape[0] if all_hidden_states else 0
        
        if total_length != actual_length:
            # Adjust last sample length to match
            diff = actual_length - total_length
            sample_lengths[-1] += diff
        
        # Split hidden states by samples
        samples_hidden_states = []
        start_idx = 0
        
        for sample_length in sample_lengths:
            if sample_length <= 0:
                continue
            
            end_idx = start_idx + sample_length
            
            # Collect all layers for this sample
            sample_all_layers = []
            for layer_hidden_states in all_hidden_states:
                sample_layer_hidden_states = layer_hidden_states[start_idx:end_idx]
                sample_all_layers.append(sample_layer_hidden_states)
            
            samples_hidden_states.append(sample_all_layers)
            start_idx = end_idx
            
            if start_idx >= actual_length:
                break
        
        return samples_hidden_states
    
    def _estimate_sample_lengths(self, outputs: Any) -> List[int]:
        """
        Estimate the length of each sample from outputs.
        
        For generate task: prompt_tokens + generated_tokens - 1
        The last generated token doesn't go through the encoder again.
        
        Args:
            outputs: vLLM RequestOutput objects
            
        Returns:
            List of sample lengths (in tokens)
        """
        if not outputs:
            return []
        
        sample_lengths = []
        for output in outputs:
            # For RequestOutput, we have prompt_token_ids and outputs
            if hasattr(output, 'prompt_token_ids'):
                prompt_length = len(output.prompt_token_ids)
                # Add generated tokens (minus 1 for the last token that doesn't encode)
                if hasattr(output, 'outputs') and output.outputs:
                    # outputs is a list of CompletionOutput
                    gen_length = len(output.outputs[0].token_ids) if output.outputs else 0
                    # The last generated token doesn't go through hidden states capture again
                    if gen_length > 0:
                        sample_lengths.append(prompt_length + gen_length - 1)
                    else:
                        sample_lengths.append(prompt_length)
                else:
                    sample_lengths.append(prompt_length)
            elif hasattr(output, 'token_ids'):
                # Fallback: use token_ids directly (assume last token not encoded)
                token_length = len(output.token_ids)
                sample_lengths.append(max(1, token_length - 1))
            else:
                # Fallback: assume 1 token
                sample_lengths.append(1)
        
        return sample_lengths


def get_all_hidden_states_generate(
    llm: Any,
    prompts: Union[List[str], List[Dict[str, Any]]],
    max_tokens: int = 1,
    split_by_samples: bool = True,
    **generate_kwargs
) -> Union[Tuple[List[List[torch.Tensor]], Any], Tuple[List[torch.Tensor], Any]]:
    """
    便捷函数：从generate任务捕获所有层的hidden states
    
    适用于不支持embed任务的模型，如：
    - Qwen-VL, Qwen2-VL (视觉语言模型)
    - LLaVA 系列
    - 其他多模态模型
    
    通过设置max_tokens=1来实现和embed任务相同的效果（仅获取prompt的hidden states）。
    
    Args:
        llm: vLLM LLM实例（不要指定task="embed"）
        prompts: 输入prompt列表，支持:
            - List[str]: 纯文本输入
            - List[Dict]: 多模态输入（包含text, image等）
        max_tokens: 生成token数（默认1，仅触发forward获取prompt的hidden states）
        split_by_samples: 是否按样本划分（默认True）
            - If True: returns List[List[Tensor]] where [sample_idx][layer_idx]
            - If False: returns List[Tensor] where [layer_idx] (concatenated)
        **generate_kwargs: 传递给llm.generate()的额外参数
        
    Returns:
        (hidden_states, outputs)
        - 如果split_by_samples=True: 
            hidden_states[sample_idx][layer_idx] 形状为 (seq_len, hidden_size)
        - 如果split_by_samples=False: 
            hidden_states[layer_idx] 形状为 (total_tokens, hidden_size)
        
    Examples:
        >>> from vllm import LLM
        >>> import easysteer.hidden_states as hs
        >>> 
        >>> # 示例1: 纯文本输入
        >>> llm = LLM(model="Qwen2.5-VL-7B-Instruct", tensor_parallel_size=1)
        >>> hidden_states, outputs = hs.get_all_hidden_states_generate(
        ...     llm, 
        ...     prompts=["What is AI?", "Hello world!"],
        ...     max_tokens=1
        ... )
        >>> print(f"Captured {len(hidden_states)} samples")
        >>> print(f"Sample 0 has {len(hidden_states[0])} layers")
        >>> print(f"Sample 0, Layer 0 shape: {hidden_states[0][0].shape}")
        >>> 
        >>> # 示例2: 多模态输入 (text + image)
        >>> # 根据vLLM的多模态API格式构造输入
        >>> multimodal_prompts = [
        ...     {
        ...         "prompt": "Describe what you see in this image.",
        ...         "multi_modal_data": {
        ...             "image": "https://example.com/image.jpg"  # 或PIL.Image对象
        ...         }
        ...     },
        ...     {
        ...         "prompt": "What is in this picture?",
        ...         "multi_modal_data": {
        ...             "image": "path/to/local/image.jpg"
        ...         }
        ...     }
        ... ]
        >>> hidden_states, outputs = hs.get_all_hidden_states_generate(
        ...     llm,
        ...     prompts=multimodal_prompts,
        ...     max_tokens=1,
        ...     split_by_samples=True
        ... )
        >>> 
        >>> # 示例3: 获取concatenated hidden states（不按样本划分）
        >>> all_layers, outputs = hs.get_all_hidden_states_generate(
        ...     llm,
        ...     prompts=["Hello", "World"],
        ...     split_by_samples=False
        ... )
        >>> print(f"Total layers: {len(all_layers)}")
        >>> print(f"Layer 0 shape (concatenated): {all_layers[0].shape}")
    """
    capture = HiddenStatesCaptureGenerate()
    return capture.get_all_hidden_states_generate(
        llm, prompts, max_tokens, split_by_samples, **generate_kwargs
    )

