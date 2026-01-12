from flask import Blueprint, request, jsonify
import os
import sys
import logging
import json
from transformers import AutoTokenizer
import re
import time

# Import vllm related modules (using pip-installed vllm)
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# Create a blueprint for inference-related endpoints
inference_bp = Blueprint('inference', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Global shared counter to ensure uniqueness across all APIs
# This counter is incremented atomically to prevent ID collisions
# between inference_api and chat_api
# Note: Starts at 1 (not 0) as some systems require positive non-zero IDs
# Max value: 2,147,483,647 (int32 limit for vLLM)
_global_id_counter = 1

def generate_unique_id():
    """
    Generate a unique positive integer ID using a simple global counter.
    
    Returns:
        int: A unique positive integer ID (1 to 2,147,483,647)
        
    Note:
        - IDs start from 1 (not 0) as required by vLLM steer vectors
        - Counter resets on process restart
        - For production use with persistence, consider adding a database-backed counter
    """
    global _global_id_counter
    
    # Get current ID and increment
    unique_id = _global_id_counter
    _global_id_counter += 1
    
    # Safety check: wrap around if we exceed int32 max (very unlikely in practice)
    if _global_id_counter > 2147483647:
        logger.warning("Global ID counter reached int32 limit, wrapping around to 1")
        _global_id_counter = 1
    
    return unique_id

def generate_unique_name(prefix="steer_vector"):
    """Generate a unique name based on current timestamp"""
    timestamp = int(time.time() * 1000000)  # Use microseconds for more precision
    return f"{prefix}_{timestamp}"

# Store active steer vector configurations
active_steer_vectors = {}

# Store LLM instances (to avoid reloading)
llm_instances = {}

# Store tokenizers (to avoid reloading)
tokenizer_cache = {}

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
    """Get or create an LLM instance"""
    # Create a unique key
    key = f"{model_path}_{gpu_devices}"
    
    if key not in llm_instances:
        try:
            # Set GPU devices environment variable
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
            
            # Calculate tensor_parallel_size
            gpu_count = len(gpu_devices.split(','))
            
            # Create LLM instance following the new vLLM API pattern
            # enable_steer_vector=True: Enables vector steering (without this, behaves like regular vLLM)
            # enforce_eager=True: Ensures reliability and stability of interventions (strongly recommended)
            # enable_chunked_prefill=False: To avoid potential issues with steering
            llm_instances[key] = LLM(
                model=model_path,
                enable_steer_vector=True,
                enforce_eager=True,
                tensor_parallel_size=gpu_count,
                enable_chunked_prefill=False
            )
            logger.info(f"Created LLM instance for model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to create LLM instance: {str(e)}")
            raise e
    
    return llm_instances[key]

def get_model_prompt(model_path, instruction):
    """Generate appropriate prompt based on model type"""
    # Check if model path contains any identifiers to determine model type
    model_path_lower = model_path.lower()
    
    # Get or create tokenizer
    if model_path not in tokenizer_cache:
        try:
            tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Loaded tokenizer for model: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_path}, using fallback template: {str(e)}")
            # If tokenizer loading fails, we'll use fallback templates
            tokenizer_cache[model_path] = None
    
    tokenizer = tokenizer_cache[model_path]
    
    # For Gemma models, use apply_chat_template
    if 'gemma' in model_path_lower:
        if tokenizer:
            messages = [
                {"role": "user", "content": instruction}
            ]
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                logger.warning(f"Failed to apply Gemma chat template: {str(e)}")
                # Fallback for Gemma models
                return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        else:
            # Fallback for Gemma models without tokenizer
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
    
    # Default fallback (generic chat template)
    else:
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": instruction}
            ]
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                # Last resort fallback
                return f"User: {instruction}\nAssistant:"
        else:
            # Simple fallback for unknown models
            return f"User: {instruction}\nAssistant:"

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
        
        # Generate unique IDs and names for this request
        baseline_id = generate_unique_id()
        baseline_name = generate_unique_name("baseline")
        steer_id = generate_unique_id()
        steer_name = generate_unique_name(data.get('steer_vector_name', 'steer_vector'))
        
        logger.info(f"Generated unique baseline ID: {baseline_id}, name: {baseline_name}")
        logger.info(f"Generated unique steer ID: {steer_id}, name: {steer_name} (user provided: {data.get('steer_vector_name')}, {data.get('steer_vector_int_id')})")
        
        # Create baseline (non-steered) request with scale=0
        baseline_request = SteerVectorRequest(
            steer_vector_name=baseline_name,
            steer_vector_int_id=baseline_id,
            steer_vector_local_path=data['steer_vector_local_path'],  # Use the same path as the steer vector
            scale=0.0,  # Zero scale = no steering
            target_layers=[0],
            algorithm="direct"  # Simple algorithm for baseline
        )
        
        # Create the actual steering vector request with unique ID and name
        steer_vector_request = SteerVectorRequest(
            steer_vector_name=steer_name,
            steer_vector_int_id=steer_id,
            steer_vector_local_path=data['steer_vector_local_path'],
            scale=data.get('scale', 1.0),
            target_layers=data.get('target_layers'),
            prefill_trigger_tokens=data.get('prefill_trigger_tokens'),
            prefill_trigger_positions=data.get('prefill_trigger_positions'),
            generate_trigger_tokens=data.get('generate_trigger_tokens'),
            debug=data.get('debug', False),
            algorithm=data.get('algorithm', 'direct')
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
        
        # Generate unique IDs and names for this request
        baseline_id = generate_unique_id()
        baseline_name = generate_unique_name("baseline_multi")
        steer_id = generate_unique_id()
        steer_name = generate_unique_name(data.get('steer_vector_name', 'multi_vector'))
        
        logger.info(f"Generated unique baseline ID: {baseline_id}, name: {baseline_name}")
        logger.info(f"Generated unique steer ID: {steer_id}, name: {steer_name} (user provided: {data.get('steer_vector_name')}, {data.get('steer_vector_int_id')})")
        
        # Get the first vector path for baseline
        first_vector_path = data['vector_configs'][0]['path'] if data['vector_configs'] else "/dummy/path.gguf"
        
        # Create baseline (non-steered) request with scale=0
        baseline_request = SteerVectorRequest(
            steer_vector_name=baseline_name,
            steer_vector_int_id=baseline_id,
            steer_vector_local_path=first_vector_path,  # Use the first vector path
            scale=0.0,  # Zero scale = no steering
            target_layers=[0],
            algorithm="direct"  # Simple algorithm for baseline
        )
        
        # Create multi-vector steer request
        vector_configs = []
        for i, vec_config in enumerate(data['vector_configs']):
            # Validate required fields
            if 'path' not in vec_config or not vec_config['path']:
                return jsonify({'error': f'Vector config {i+1} is missing path field'}), 400
                
            # Create VectorConfig object
            from vllm.steer_vectors.request import VectorConfig
            vector_config = VectorConfig(
                path=vec_config['path'],
                scale=vec_config.get('scale', 1.0),
                target_layers=vec_config.get('target_layers'),
                prefill_trigger_positions=vec_config.get('prefill_trigger_positions', [-1]),
                prefill_trigger_tokens=vec_config.get('prefill_trigger_tokens', [-1]),  # 添加这个重要参数
                generate_trigger_tokens=vec_config.get('generate_trigger_tokens', [-1]),  # 添加这个重要参数
                algorithm=vec_config.get('algorithm', 'direct'),
                normalize=vec_config.get('normalize', False)
            )
            vector_configs.append(vector_config)
        
        # Create the multi-vector steering request with unique ID and name
        steer_vector_request = SteerVectorRequest(
            steer_vector_name=steer_name,
            steer_vector_int_id=steer_id,
            vector_configs=vector_configs,
            debug=data.get('debug', False),
            conflict_resolution=data.get('conflict_resolution', 'sequential')
        )
        
        # 记录详细的向量配置信息，用于调试
        logger.info(f"Multi-vector request details:")
        logger.info(f"- Model path: {data['model_path']}")
        logger.info(f"- Vector name: {data['steer_vector_name']}")
        logger.info(f"- Vector ID: {data['steer_vector_int_id']}")
        logger.info(f"- Conflict resolution: {data.get('conflict_resolution', 'sequential')}")
        logger.info(f"- Number of vectors: {len(vector_configs)}")
        
        for i, vec_config in enumerate(vector_configs):
            logger.info(f"Vector {i+1} details:")
            logger.info(f"- Path: {vec_config.path}")
            logger.info(f"- Scale: {vec_config.scale}")
            logger.info(f"- Algorithm: {vec_config.algorithm}")
            logger.info(f"- Target layers: {vec_config.target_layers}")
            logger.info(f"- Prefill trigger tokens: {vec_config.prefill_trigger_tokens}")
            logger.info(f"- Prefill trigger positions: {vec_config.prefill_trigger_positions}")
            logger.info(f"- Generate trigger tokens: {vec_config.generate_trigger_tokens}")
            logger.info(f"- Normalize: {vec_config.normalize}")
        
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
            logger.info(f"Generated multi-vector text comparison with {len(vector_configs)} vectors")
            
            # 构建包含更详细配置信息的响应
            response = {
                'success': True,
                'baseline_text': baseline_text,  # Unsteered output
                'generated_text': steered_text,  # Steered output
                'prompt': prompt,
                'config': {
                    'model_path': data['model_path'],
                    'steer_vector_name': steer_vector_request.steer_vector_name,
                    'num_vectors': len(vector_configs),
                    'conflict_resolution': data.get('conflict_resolution', 'sequential'),
                    'vectors': [
                        {
                            'path': vec.path,
                            'scale': vec.scale,
                            'algorithm': vec.algorithm,
                            'target_layers': vec.target_layers,
                            'prefill_trigger_tokens': vec.prefill_trigger_tokens,
                            'prefill_trigger_positions': vec.prefill_trigger_positions,
                            'generate_trigger_tokens': vec.generate_trigger_tokens
                        }
                        for vec in vector_configs
                    ]
                }
            }
            
            logger.info(f"Generated multi-vector text comparison with {len(vector_configs)} vectors")
            
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
    """Fully restart the backend process with proper GPU memory cleanup"""
    try:
        import sys
        import threading
        import time
        import gc
        
        logger.info("Preparing to fully restart the backend process...")
        
        def delayed_restart():
            """Delayed restart to allow response to be sent"""
            time.sleep(1)  # Wait 1 second for the response to be sent
            logger.info("Restarting backend process...")
            
            # Step 1: Clear all LLM instances from inference_api
            logger.info("Cleaning up inference LLM instances...")
            global llm_instances
            for key in list(llm_instances.keys()):
                try:
                    logger.info(f"Deleting LLM instance: {key}")
                    del llm_instances[key]
                except Exception as e:
                    logger.error(f"Failed to delete LLM instance {key}: {str(e)}")
            llm_instances.clear()
            
            # Step 2: Clear LLM instances from chat_api if available
            try:
                from chat_api import chat_llm_instances
                logger.info("Cleaning up chat LLM instances...")
                for key in list(chat_llm_instances.keys()):
                    try:
                        logger.info(f"Deleting chat LLM instance: {key}")
                        del chat_llm_instances[key]
                    except Exception as e:
                        logger.error(f"Failed to delete chat LLM instance {key}: {str(e)}")
                chat_llm_instances.clear()
            except ImportError:
                logger.info("chat_api module not available, skipping chat LLM cleanup")
            except Exception as e:
                logger.error(f"Error cleaning up chat LLM instances: {str(e)}")
            
            # Step 3: Clear tokenizer cache
            logger.info("Cleaning up tokenizer cache...")
            global tokenizer_cache
            tokenizer_cache.clear()
            
            # Step 4: Force garbage collection
            logger.info("Running garbage collection...")
            gc.collect()
            
            # Step 5: Clear GPU memory cache (if torch is available)
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("Clearing CUDA cache...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Log GPU memory status
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        logger.info(f"GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            except ImportError:
                logger.info("PyTorch not available, skipping CUDA cache cleanup")
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {str(e)}")
            
            # Step 6: Get current Python executable and script arguments
            python_executable = sys.executable
            script_args = sys.argv
            
            # Step 7: Use os.execv to restart the process
            logger.info("Executing process restart...")
            import os
            os.execv(python_executable, [python_executable] + script_args)
        
        # Execute restart in a new thread to avoid blocking the response
        restart_thread = threading.Thread(target=delayed_restart)
        restart_thread.daemon = True
        restart_thread.start()
        
        return jsonify({
            "success": True,
            "message": "Backend is restarting and cleaning up GPU memory, please try again in a few seconds..."
        })
    
    except Exception as e:
        logger.error(f"Failed to restart backend: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Failed to restart backend: {str(e)}"
        }), 500 