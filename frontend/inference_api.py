from flask import Blueprint, request, jsonify
import os
import sys
import logging
import json
import re
import time

# Import vllm related modules (using pip-installed vllm)
from vllm import SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# Import core modules for unified management
from core import (
    generate_unique_id, 
    generate_unique_name, 
    llm_manager, 
    resource_manager,
    prompt_formatter,
    SteerRequestBuilder
)

# Create a blueprint for inference-related endpoints
inference_bp = Blueprint('inference', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Store active steer vector configurations
active_steer_vectors = {}

# Keep backward compatibility: create references to old global variables
# These will be accessed by resource_manager for cleanup
llm_instances = llm_manager._instances  # Reference to LLM manager's internal cache
tokenizer_cache = prompt_formatter._tokenizer_cache  # Reference to prompt formatter's cache

def get_message(key, lang='zh', **kwargs):
    """Get a message in the specified language"""
    error_messages = {
        'zh': {
            'missing_field': '缺少必填字段: {field}',
            'file_not_found': '文件不存在: {path}',
            'server_error': '服务器错误: {error}',
            'not_found': 'Steer Vector ID {id} 不存在',
            'deleted': 'Steer Vector {name} 已删除',
            'created': 'Steer Vector配置创建成功',
            'generation_error': '生成失败: {error}',
            'model_loading_error': '模型加载失败: {error}'
        },
        'en': {
            'missing_field': 'Missing required field: {field}',
            'file_not_found': 'File not found: {path}',
            'server_error': 'Server error: {error}',
            'not_found': 'Steer Vector ID {id} does not exist',
            'deleted': 'Steer Vector {name} has been deleted',
            'created': 'Steer Vector configuration created successfully',
            'generation_error': 'Generation failed: {error}',
            'model_loading_error': 'Model loading failed: {error}'
        }
    }
    messages = error_messages.get(lang, error_messages['zh'])
    message = messages.get(key, key)
    return message.format(**kwargs)

def get_or_create_llm(model_path, gpu_devices):
    """
    Get or create an LLM instance (wrapper for backward compatibility).
    
    This function wraps the core LLMManager to maintain API compatibility.
    """
    return llm_manager.get_or_create_llm(
        model_path=model_path,
        gpu_devices=gpu_devices,
        enable_steer_vector=True,
        enforce_eager=True,
        enable_chunked_prefill=False
    )

def get_model_prompt(model_path, instruction):
    """
    Generate appropriate prompt based on model type (wrapper for backward compatibility).
    
    This function wraps the core PromptFormatter to maintain API compatibility.
    """
    return prompt_formatter.format_single_turn(model_path, instruction)

@inference_bp.route('/api/generate', methods=['POST'])
def generate():
    """Generate text using a Steer Vector with baseline comparison"""
    try:
        data = request.json
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        
        # Validate required fields
        required_fields = ['model_path', 'instruction', 'steer_vector_name', 'steer_vector_int_id', 'steer_vector_local_path']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': get_message('missing_field', lang, field=field)}), 400
        
        # Get or create LLM instance
        try:
            llm = get_or_create_llm(
                data['model_path'],
                data.get('gpu_devices', '0')
            )
        except Exception as e:
            return jsonify({'error': get_message('model_loading_error', lang, error=str(e))}), 500
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=data.get('sampling_params', {}).get('temperature', 0.0),
            max_tokens=data.get('sampling_params', {}).get('max_tokens', 128),
            repetition_penalty=data.get('sampling_params', {}).get('repetition_penalty', 1.1)
        )
        
        # Format input based on model type
        prompt = get_model_prompt(data['model_path'], data['instruction'])
        
        # Get algorithm and target_layers for baseline request
        algorithm = data.get('algorithm', 'direct')
        target_layers = data.get('target_layers')
        
        # Create baseline (non-steered) request using builder
        # Must use same algorithm and target_layers as the actual vector to load it correctly
        baseline_request = SteerRequestBuilder.build_baseline_request(
            vector_path=data['steer_vector_local_path'],
            algorithm=algorithm,
            target_layers=target_layers
        )
        
        # Create the actual steering vector request using builder
        steer_vector_request = SteerRequestBuilder.build_single_vector_request(
            vector_path=data['steer_vector_local_path'],
            scale=data.get('scale', 1.0),
            target_layers=target_layers,
            algorithm=algorithm,
            steer_name=data.get('steer_vector_name'),
            prefill_trigger_tokens=data.get('prefill_trigger_tokens'),
            prefill_trigger_positions=data.get('prefill_trigger_positions'),
            generate_trigger_tokens=data.get('generate_trigger_tokens'),
            debug=data.get('debug', False)
        )
        
        try:
            # First, generate baseline (non-steered) output
            baseline_output = llm.generate(
                prompt,
                steer_vector_request=baseline_request,
                sampling_params=sampling_params
            )
            baseline_text = baseline_output[0].outputs[0].text
            
            # Then generate steered output
            steered_output = llm.generate(
                prompt,
                steer_vector_request=steer_vector_request,
                sampling_params=sampling_params
            )
            steered_text = steered_output[0].outputs[0].text
            
            # Return success response with both outputs
            response = {
                'success': True,
                'baseline_text': baseline_text,  # Unsteered output
                'generated_text': steered_text,  # Steered output
                'prompt': prompt,
                'config': {
                    'model_path': data['model_path'],
                    'steer_vector_name': steer_vector_request.steer_vector_name,
                    'algorithm': steer_vector_request.algorithm,
                    'scale': steer_vector_request.scale,
                    'target_layers': steer_vector_request.target_layers
                }
            }
            
            logger.info(f"Generated text comparison with steer vector: {steer_vector_request.steer_vector_name}")
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return jsonify({'error': get_message('generation_error', lang, error=str(e))}), 500
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        return jsonify({'error': get_message('server_error', lang, error=str(e))}), 500

@inference_bp.route('/api/generate-multi', methods=['POST'])
def generate_multi():
    """Generate text using multiple Steer Vectors with baseline comparison"""
    try:
        data = request.json
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        
        # Validate required fields
        required_fields = ['model_path', 'instruction', 'steer_vector_name', 'steer_vector_int_id', 'vector_configs']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': get_message('missing_field', lang, field=field)}), 400
        
        # Validate vector configs
        if not isinstance(data['vector_configs'], list) or len(data['vector_configs']) == 0:
            return jsonify({'error': get_message('missing_field', lang, field='vector_configs (should be non-empty array)')}), 400
        
        # Get or create LLM instance
        try:
            llm = get_or_create_llm(
                data['model_path'],
                data.get('gpu_devices', '0')
            )
        except Exception as e:
            return jsonify({'error': get_message('model_loading_error', lang, error=str(e))}), 500
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=data.get('sampling_params', {}).get('temperature', 0.0),
            max_tokens=data.get('sampling_params', {}).get('max_tokens', 128),
            repetition_penalty=data.get('sampling_params', {}).get('repetition_penalty', 1.1)
        )
        
        # Format input based on model type
        prompt = get_model_prompt(data['model_path'], data['instruction'])
        
        # Get the first vector configuration for baseline
        first_vector_config = data['vector_configs'][0] if data['vector_configs'] else {}
        first_vector_path = first_vector_config.get('path', "/dummy/path.gguf")
        first_algorithm = first_vector_config.get('algorithm', 'direct')
        first_target_layers = first_vector_config.get('target_layers')
        
        # Create baseline (non-steered) request using builder
        # Must use same algorithm and target_layers as the first vector to load it correctly
        baseline_request = SteerRequestBuilder.build_baseline_request(
            vector_path=first_vector_path,
            algorithm=first_algorithm,
            target_layers=first_target_layers
        )
        
        # Create multi-vector steer request using builder
        steer_vector_request = SteerRequestBuilder.build_multi_vector_request(
            vector_configs=data['vector_configs'],
            conflict_resolution=data.get('conflict_resolution', 'sequential'),
            steer_name=data.get('steer_vector_name'),
            debug=data.get('debug', False)
        )
        
        try:
            # First, generate baseline (non-steered) output
            baseline_output = llm.generate(
                prompt,
                steer_vector_request=baseline_request,
                sampling_params=sampling_params
            )
            baseline_text = baseline_output[0].outputs[0].text
            
            # Then generate steered output with multiple vectors
            steered_output = llm.generate(
                prompt,
                steer_vector_request=steer_vector_request,
                sampling_params=sampling_params
            )
            steered_text = steered_output[0].outputs[0].text
            
            # 记录生成结果并返回更详细的配置信息
            num_vectors = len(data['vector_configs'])
            logger.info(f"Generated multi-vector text comparison with {num_vectors} vectors")
            
            # 构建包含更详细配置信息的响应
            response = {
                'success': True,
                'baseline_text': baseline_text,  # Unsteered output
                'generated_text': steered_text,  # Steered output
                'prompt': prompt,
                'config': {
                    'model_path': data['model_path'],
                    'steer_vector_name': steer_vector_request.steer_vector_name,
                    'num_vectors': num_vectors,
                    'conflict_resolution': data.get('conflict_resolution', 'sequential'),
                    'vectors': data['vector_configs']
                }
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return jsonify({'error': get_message('generation_error', lang, error=str(e))}), 500
        
    except Exception as e:
        logger.error(f"Error in generate-multi endpoint: {str(e)}")
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        return jsonify({'error': get_message('server_error', lang, error=str(e))}), 500

@inference_bp.route('/api/config/<config_name>', methods=['GET'])
def get_config(config_name):
    """Get a configuration file"""
    try:
        # 首先检查单向量配置目录
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'inference', f'{config_name}.json')
        
        # 如果单向量目录中没找到，检查多向量配置目录
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(__file__), 'configs', 'multi_vector', f'{config_name}.json')
            
        if not os.path.exists(config_path):
            return jsonify({"error": f"Configuration {config_name} not found"}), 404
        
        # 读取并返回配置
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # 如果是多向量配置，添加一个标识
        if 'vector_configs' in config:
            config['is_multi_vector'] = True
            
        return jsonify(config)
    
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        return jsonify({"error": f"Failed to get configuration: {str(e)}"}), 500

@inference_bp.route('/api/configs', methods=['GET'])
def list_configs():
    """List all available configuration files"""
    try:
        # 获取单向量配置
        single_vector_configs_dir = os.path.join(os.path.dirname(__file__), 'configs', 'inference')
        multi_vector_configs_dir = os.path.join(os.path.dirname(__file__), 'configs', 'multi_vector')
        
        # 定义配置友好名称
        config_display_names = {
            'emoji_loreft': 'Emoji LoReft Configuration',
            'emotion_direct': 'Emotion Direct Configuration',
            'refusal_direction': 'Refusal Direction Control'
        }
        
        configs = []
        
        # 处理单向量配置
        if os.path.exists(single_vector_configs_dir):
            for filename in os.listdir(single_vector_configs_dir):
                if filename.endswith('.json'):
                    config_name = filename[:-5]  # 去除.json后缀
                    display_name = config_display_names.get(config_name, config_name.replace('_', ' ').title())
                    configs.append({
                        "name": config_name,
                        "display_name": display_name,
                        "type": "single_vector"
                    })
        
        # 处理多向量配置
        if os.path.exists(multi_vector_configs_dir):
            for filename in os.listdir(multi_vector_configs_dir):
                if filename.endswith('.json'):
                    config_name = filename[:-5]  # 去除.json后缀
                    display_name = config_display_names.get(config_name, config_name.replace('_', ' ').title())
                    configs.append({
                        "name": config_name,
                        "display_name": display_name,
                        "type": "multi_vector"
                    })
        
        # 按名称排序
        configs.sort(key=lambda x: x['display_name'])
        
        return jsonify({"configs": configs})
    
    except Exception as e:
        logger.error(f"Error listing configs: {str(e)}")
        return jsonify({"error": f"Failed to list configurations: {str(e)}"}), 500

@inference_bp.route('/api/steer-vector', methods=['POST'])
def create_steer_vector():
    """Create or update a Steer Vector configuration (kept for config management)"""
    try:
        data = request.json
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        
        # Validate required fields (user still needs to provide these, but they will be replaced)
        required_fields = ['steer_vector_name', 'steer_vector_int_id', 'steer_vector_local_path']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': get_message('missing_field', lang, field=field)}), 400
        
        # Generate unique ID and name (replace user-provided values)
        unique_id = generate_unique_id()
        unique_name = generate_unique_name(data['steer_vector_name'])
        
        logger.info(f"Generated unique ID: {unique_id}, name: {unique_name} (user provided: {data['steer_vector_name']}, {data['steer_vector_int_id']})")
        
        # Create SteerVectorRequest object with unique ID and name
        steer_vector_request = SteerVectorRequest(
            steer_vector_name=unique_name,
            steer_vector_int_id=unique_id,
            steer_vector_local_path=data['steer_vector_local_path'],
            scale=data.get('scale', 1.0),
            target_layers=data.get('target_layers'),
            prefill_trigger_tokens=data.get('prefill_trigger_tokens'),
            prefill_trigger_positions=data.get('prefill_trigger_positions'),
            generate_trigger_tokens=data.get('generate_trigger_tokens'),
            debug=data.get('debug', False),
            algorithm=data.get('algorithm', 'direct')
        )
        
        # Validate if file exists
        if not os.path.exists(steer_vector_request.steer_vector_local_path):
            return jsonify({'error': get_message('file_not_found', lang, path=steer_vector_request.steer_vector_local_path)}), 400
        
        # Store configuration
        active_steer_vectors[steer_vector_request.steer_vector_int_id] = steer_vector_request
        
        # Return success response
        response = {
            'success': True,
            'message': get_message('created', lang),
            'steer_vector_int_id': steer_vector_request.steer_vector_int_id,
            'config': {
                'name': steer_vector_request.steer_vector_name,
                'id': steer_vector_request.steer_vector_int_id,
                'path': steer_vector_request.steer_vector_local_path,
                'scale': steer_vector_request.scale,
                'algorithm': steer_vector_request.algorithm,
                'target_layers': steer_vector_request.target_layers,
                'prefill_trigger_tokens': steer_vector_request.prefill_trigger_tokens,
                'prefill_trigger_positions': steer_vector_request.prefill_trigger_positions,
                'generate_trigger_tokens': steer_vector_request.generate_trigger_tokens,
                'debug': steer_vector_request.debug
            }
        }
        
        logger.info(f"Created steer vector: {steer_vector_request.steer_vector_name} (ID: {steer_vector_request.steer_vector_int_id})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error creating steer vector: {str(e)}")
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        return jsonify({'error': get_message('server_error', lang, error=str(e))}), 500

@inference_bp.route('/api/steer-vector/<int:steer_vector_int_id>', methods=['GET'])
def get_steer_vector(steer_vector_int_id):
    """Get a specific Steer Vector configuration"""
    lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
    
    if steer_vector_int_id in active_steer_vectors:
        sv = active_steer_vectors[steer_vector_int_id]
        return jsonify({
            'success': True,
            'config': {
                'name': sv.steer_vector_name,
                'id': sv.steer_vector_int_id,
                'path': sv.steer_vector_local_path,
                'scale': sv.scale,
                'algorithm': sv.algorithm,
                'target_layers': sv.target_layers,
                'prefill_trigger_tokens': sv.prefill_trigger_tokens,
                'prefill_trigger_positions': sv.prefill_trigger_positions,
                'generate_trigger_tokens': sv.generate_trigger_tokens,
                'debug': sv.debug
            }
        }), 200
    else:
        return jsonify({'error': get_message('not_found', lang, id=steer_vector_int_id)}), 404

@inference_bp.route('/api/steer-vectors', methods=['GET'])
def list_steer_vectors():
    """List all active Steer Vector configurations"""
    vectors = []
    for sv_id, sv in active_steer_vectors.items():
        vectors.append({
            'id': sv.steer_vector_int_id,
            'name': sv.steer_vector_name,
            'algorithm': sv.algorithm,
            'scale': sv.scale
        })
    
    return jsonify({
        'success': True,
        'count': len(vectors),
        'steer_vectors': vectors
    }), 200

@inference_bp.route('/api/steer-vector/<int:steer_vector_int_id>', methods=['DELETE'])
def delete_steer_vector(steer_vector_int_id):
    """Delete a Steer Vector configuration"""
    lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
    
    if steer_vector_int_id in active_steer_vectors:
        sv_name = active_steer_vectors[steer_vector_int_id].steer_vector_name
        del active_steer_vectors[steer_vector_int_id]
        logger.info(f"Deleted steer vector: {sv_name} (ID: {steer_vector_int_id})")
        return jsonify({
            'success': True,
            'message': get_message('deleted', lang, name=sv_name)
        }), 200
    else:
        return jsonify({'error': get_message('not_found', lang, id=steer_vector_int_id)}), 404

@inference_bp.route('/api/restart', methods=['POST'])
def restart_backend():
    """
    Fully restart the backend process with proper GPU memory cleanup.
    
    This endpoint uses the unified ResourceManager for cleanup and restart.
    """
    try:
        result = resource_manager.restart_backend(delay=1.0)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to restart backend: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Failed to restart backend: {str(e)}"
        }), 500 