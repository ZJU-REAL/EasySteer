# SPDX-License-Identifier: Apache-2.0
"""
MoE Router Logits Capture for Generate Task

Extends MoE capture to work with generate task, not just embed.
"""

from typing import Any, Dict, Optional, List, Union, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


class MoERouterLogitsCaptureGenerate:
    """
    MoE router logits capture for generate task.
    
    This works with llm.generate() instead of llm.embed().
    """
    
    def __init__(self):
        """Initialize generate-mode MoE capture."""
        pass
    
    def get_router_logits_generate(
        self, 
        llm: Any, 
        prompts: Union[List[str], List[Dict[str, Any]]],
        max_tokens: int = 1,
        split_by_samples: bool = False,
        **generate_kwargs
    ) -> Union[Tuple[Dict[int, torch.Tensor], Any], Tuple[List[Dict[int, torch.Tensor]], Any]]:
        """
        Get router logits from MoE layers using generate task.
        
        适用于不支持embed任务的模型（如Qwen3-VL）。支持多模态输入。
        
        Args:
            llm: The vLLM LLM instance (must NOT specify task="embed")
            prompts: List of input prompts,支持以下格式:
                - List[str]: 纯文本输入
                - List[Dict]: 多模态输入，例如包含text和image
            max_tokens: Number of tokens to generate (default 1, just to trigger forward)
            split_by_samples: Whether to split router logits by samples (default: False)
                - If True: returns List[Dict[layer_id, Tensor]] where [sample_idx][layer_id]
                - If False: returns Dict[layer_id, Tensor] (concatenated all samples)
            **generate_kwargs: Additional arguments passed to llm.generate()
                (e.g., temperature, top_p, etc.)
            
        Returns:
            Tuple of (router_logits, outputs):
            - If split_by_samples=False:
                router_logits: Dict[layer_id, tensor(num_tokens, n_experts)]
            - If split_by_samples=True:
                router_logits: List[Dict[layer_id, tensor(seq_len, n_experts)]]
                    router_logits[sample_idx][layer_id] for each sample
            - outputs: vLLM generation outputs
            
        Examples:
            >>> from vllm import LLM
            >>> import easysteer.hidden_states as hs
            >>> 
            >>> # 文本输入
            >>> llm = LLM(model="Qwen3-VL-30B-A3B-Thinking", tensor_parallel_size=4)
            >>> router_logits, outputs = hs.get_moe_router_logits_generate(
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
            >>> router_logits, outputs = hs.get_moe_router_logits_generate(
            ...     llm,
            ...     prompts=multimodal_prompts,
            ...     max_tokens=10
            ... )
        """
        try:
            # Step 1: Enable MoE capture via RPC
            self._enable_capture(llm)
            
            # Step 2: Run generation (只生成1个token来触发forward)
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
            
            # Step 3: Get captured router logits via RPC
            router_logits = self._get_captured_logits(llm)
            
            # Step 4: Split by samples if requested
            if split_by_samples:
                samples_router_logits = self._split_router_logits_by_samples(
                    router_logits, outputs
                )
                return samples_router_logits, outputs
            else:
                return router_logits, outputs
                
        finally:
            # Step 5: Always cleanup
            self._cleanup(llm)
    
    def _enable_capture(self, llm: Any) -> None:
        """Enable MoE router logits capture via RPC."""
        llm.llm_engine.engine_core.collective_rpc("enable_moe_router_logits_capture")
    
    def _get_captured_logits(self, llm: Any) -> Dict[int, torch.Tensor]:
        """Get captured router logits from workers via RPC."""
        # Import deserialization utility
        from vllm.hidden_states import deserialize_moe_router_logits
        
        # Call RPC to get serialized router logits
        results = llm.llm_engine.engine_core.collective_rpc("get_moe_router_logits")
        
        # Deserialize from dict format to tensors
        router_logits_dict = deserialize_moe_router_logits(results[0])
        
        return router_logits_dict
    
    def _cleanup(self, llm: Any) -> None:
        """Cleanup: disable capture and clear memory."""
        try:
            llm.llm_engine.engine_core.collective_rpc("clear_moe_router_logits")
            llm.llm_engine.engine_core.collective_rpc("disable_moe_router_logits_capture")
        except Exception:
            pass
    
    def _split_router_logits_by_samples(
        self,
        all_router_logits: Dict[int, torch.Tensor],
        outputs: Any
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Split concatenated router logits by samples.
        
        Args:
            all_router_logits: Dict[layer_id, Tensor], each shape (total_tokens, n_experts)
            outputs: vLLM RequestOutput objects
            
        Returns:
            List of dicts: [sample_idx][layer_id], each shape (seq_len, n_experts)
        """
        if not all_router_logits or not outputs:
            return []
        
        # Get sample lengths from outputs
        sample_lengths = self._estimate_sample_lengths(outputs)
        
        if not sample_lengths:
            # Fallback: return all as one sample
            return [all_router_logits]
        
        # Verify total length matches
        total_length = sum(sample_lengths)
        first_layer_id = next(iter(all_router_logits.keys()))
        actual_length = all_router_logits[first_layer_id].shape[0]
        
        if total_length != actual_length:
            # Adjust last sample length to match
            diff = actual_length - total_length
            sample_lengths[-1] += diff
        
        # Split router logits by samples
        samples_router_logits = []
        start_idx = 0
        
        for sample_length in sample_lengths:
            if sample_length <= 0:
                continue
            
            end_idx = start_idx + sample_length
            
            # Extract this sample's logits for all layers
            sample_logits = {}
            for layer_id, logits_tensor in all_router_logits.items():
                sample_logits[layer_id] = logits_tensor[start_idx:end_idx]
            
            samples_router_logits.append(sample_logits)
            start_idx = end_idx
            
            if start_idx >= actual_length:
                break
        
        return samples_router_logits
    
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
                    # The last generated token doesn't go through MoE layers again
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


def get_moe_router_logits_generate(
    llm: Any,
    prompts: Union[List[str], List[Dict[str, Any]]],
    max_tokens: int = 1,
    split_by_samples: bool = False,
    **generate_kwargs
) -> Union[Tuple[Dict[int, torch.Tensor], Any], Tuple[List[Dict[int, torch.Tensor]], Any]]:
    """
    便捷函数：从generate任务捕获MoE router logits
    
    适用于不支持embed任务的模型，如：
    - Qwen3-VL (视觉语言模型)
    - 其他多模态MoE模型
    
    支持纯文本和多模态输入。
    
    Args:
        llm: vLLM LLM实例（不要指定task="embed"）
        prompts: 输入prompt列表，支持:
            - List[str]: 纯文本输入
            - List[Dict]: 多模态输入（包含text, image等）
        max_tokens: 生成token数（默认1，仅触发forward）
        split_by_samples: 是否按样本划分（默认False）
        **generate_kwargs: 传递给llm.generate()的额外参数
        
    Returns:
        (router_logits, outputs)
        - 如果split_by_samples=False: router_logits是Dict[layer_id, Tensor]
        - 如果split_by_samples=True: router_logits是List[Dict[layer_id, Tensor]]
        
    Examples:
        >>> from vllm import LLM
        >>> import easysteer.hidden_states as hs
        >>> 
        >>> # 示例1: 纯文本输入
        >>> llm = LLM(model="Qwen3-VL-30B-A3B-Thinking", tensor_parallel_size=4)
        >>> router_logits, outputs = hs.get_moe_router_logits_generate(
        ...     llm, 
        ...     prompts=["What is AI?"],
        ...     max_tokens=1
        ... )
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
        >>> router_logits, outputs = hs.get_moe_router_logits_generate(
        ...     llm,
        ...     prompts=multimodal_prompts,
        ...     max_tokens=10,
        ...     split_by_samples=True
        ... )
        >>> 
        >>> # 示例3: 查看结果
        >>> for layer_id, logits in router_logits.items():
        ...     print(f"Layer {layer_id}: {logits.shape}")
    """
    capture = MoERouterLogitsCaptureGenerate()
    return capture.get_router_logits_generate(
        llm, prompts, max_tokens, split_by_samples, **generate_kwargs
    )

