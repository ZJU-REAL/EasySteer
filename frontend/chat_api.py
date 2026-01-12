from flask import Blueprint, request, jsonify
import logging
import time
import random
import json
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import vllm related modules
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# Create blueprint
chat_bp = Blueprint('chat', __name__)

# Import the shared ID generation functions from inference_api
# This ensures all APIs use the same counter and avoid ID collisions
import sys
import os
# Add the frontend directory to the path if not already there
frontend_dir = os.path.dirname(os.path.abspath(__file__))
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

# Try to import from inference_api, if it fails, define locally
try:
    from inference_api import generate_unique_id, generate_unique_name
    logger.info("Successfully imported shared ID generation functions from inference_api")
except ImportError:
    logger.warning("Could not import from inference_api, defining local ID generation functions")
    # Fallback: define local functions
    _chat_id_counter = 1  # Start from 1 (not 0)
    
    def generate_unique_id():
        """
        Generate a unique positive integer ID using a simple global counter.
        
        Returns:
            int: A unique positive integer ID (1 to 2,147,483,647)
        """
        global _chat_id_counter
        
        # Get current ID and increment
        unique_id = _chat_id_counter
        _chat_id_counter += 1
        
        # Safety check: wrap around if we exceed int32 max
        if _chat_id_counter > 2147483647:
            logger.warning("Chat ID counter reached int32 limit, wrapping around to 1")
            _chat_id_counter = 1
        
        return unique_id
    
    def generate_unique_name(prefix="steer_vector"):
        """Generate a unique name based on current timestamp"""
        timestamp = int(time.time() * 1000000)  # Use microseconds for more precision
        return f"{prefix}_{timestamp}"

# Store LLM instances (to avoid reloading)
chat_llm_instances = {}

# Placeholder for models/vectors/etc.
presets = {
    "happy_mode": {"name": "Happy Mode", "description": "Responds in a cheerful and positive manner"},
    "chinese": {"name": "Chinese Mode", "description": "Responds in Chinese language"},
    "reject_mode": {"name": "Reject Mode", "description": "Rejects inappropriate requests"},
    "cat_mode": {"name": "Cat Mode", "description": "Responds like a cat"}
}

# Explicitly map preset keys to their config files
PRESET_CONFIG_PATHS = {
    "happy_mode": "configs/chat/happy_mode.json",
    "chinese": "configs/chat/chinese_mode.json",
    "reject_mode": "configs/chat/reject_mode.json",
    "cat_mode": "configs/chat/cat_mode.json"
}

# Preset configurations
preset_configs = {}

def load_preset_configs():
    """Load preset configurations from the explicit paths defined in PRESET_CONFIG_PATHS."""
    base_dir = os.path.dirname(__file__)
    for preset_name, config_path_str in PRESET_CONFIG_PATHS.items():
        try:
            config_path = os.path.join(base_dir, config_path_str)
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Store the loaded config
            preset_configs[preset_name] = {
                "vector_path": config["vector"]["path"],
                "scale": config["vector"]["scale"],
                "target_layers": config["vector"]["target_layers"],
                "algorithm": config["vector"]["algorithm"],
                "prefill_trigger_token_ids": config["vector"]["prefill_trigger_token_ids"],
                "generate_trigger_token_ids": config["vector"].get("generate_trigger_token_ids", None),  # Safely get this key
                "normalize": config["vector"].get("normalize", False),  # Safely get this key, default to False
                "model_path": config["model"]["path"]
            }
            logger.info(f"Successfully loaded config for preset: {preset_name} from {config_path_str}")
        except Exception as e:
            logger.error(f"Failed to load config file {config_path_str} for preset {preset_name}: {str(e)}")

def get_or_create_llm(model_path, gpu_devices="0"):
    """Get or create an LLM instance"""
    # Create a unique key
    key = f"{model_path}_{gpu_devices}"
    
    if key not in chat_llm_instances:
        try:
            # Set GPU devices environment variable
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
            
            # Calculate tensor_parallel_size
            gpu_count = len(gpu_devices.split(','))
            
            # Create LLM instance following the new vLLM API pattern
            # enable_steer_vector=True: Enables vector steering (without this, behaves like regular vLLM)
            # enforce_eager=True: Ensures reliability and stability of interventions (strongly recommended)
            # enable_chunked_prefill=False: To avoid potential issues with steering
            chat_llm_instances[key] = LLM(
                model=model_path,
                enable_steer_vector=True,
                enforce_eager=True,
                tensor_parallel_size=gpu_count,
                enable_chunked_prefill=False
            )
            logger.info(f"Created LLM instance for chat model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to create LLM instance: {str(e)}")
            raise e
    
    return chat_llm_instances[key]

def get_model_prompt(model_path, message, history=None):
    """Generate appropriate prompt based on model type and include conversation history"""
    model_path_lower = model_path.lower()
    prompt = ""
    
    # 处理历史对话
    if history and len(history) > 0:
        for turn in history:
            if "gemma" in model_path_lower:
                if turn.get("role") == "user":
                    prompt += f"<start_of_turn>user\n{turn.get('content')}<end_of_turn>\n"
                else:
                    prompt += f"<start_of_turn>model\n{turn.get('content')}<end_of_turn>\n"
            elif "qwen" in model_path_lower:
                if turn.get("role") == "user":
                    prompt += f"<|im_start|>user\n{turn.get('content')}<|im_end|>\n"
                else:
                    prompt += f"<|im_start|>assistant\n{turn.get('content')}<|im_end|>\n"
            else:
                # 通用格式
                if turn.get("role") == "user":
                    prompt += f"User: {turn.get('content')}\n"
                else:
                    prompt += f"Assistant: {turn.get('content')}\n"
    
    # 添加当前消息
    if "gemma" in model_path_lower:
        prompt += f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model"
    elif "qwen" in model_path_lower:
        prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant"
    else:
        prompt += f"User: {message}\nAssistant:"
    
    return prompt

@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat API endpoint - processes a chat request and returns a response
    """
    try:
        data = request.json
        logger.info(f"Chat request received: {data}")
        
        # Extract parameters
        preset = data.get('preset', 'happy_mode')
        message = data.get('message', '')
        history = data.get('history', [])  # 获取普通对话历史
        steered_history = data.get('steered_history', [])  # 获取引导对话历史
        gpu_devices = data.get('gpu_devices', '0')
        temperature = float(data.get('temperature', 0.8))
        max_tokens = int(data.get('max_tokens', 512))
        repetition_penalty = float(data.get('repetition_penalty', 1.1))
        
        # Check if we have config for this preset
        if preset not in preset_configs:
            logger.warning(f"No config found for preset: {preset}. Using dummy response.")
            
            # Simulate a delay and return dummy responses
            time.sleep(0.5)
            
            normal_response = f"This is a normal response to: {message}"
            steered_response = f"This is a steered response ({preset}) to: {message}"
            
            response = {
                'success': True,
                'normal_response': normal_response,
                'steered_response': steered_response,
                'preset': preset
            }
            
            return jsonify(response)
        
        # Get config for the preset
        config = preset_configs[preset]
        model_path = config["model_path"]
        print(model_path)
        # Get or create LLM
        try:
            llm = get_or_create_llm(model_path, gpu_devices)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Failed to load model: {str(e)}"
            }), 500
        
        # Format prompt
        prompt = get_model_prompt(model_path, message, history)
        # Format prompt for steered response (using the same history for now)
        steered_prompt = get_model_prompt(model_path, message, steered_history)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )
        
        # Generate unique IDs and names for this chat request
        baseline_id = generate_unique_id()
        baseline_name = generate_unique_name("chat_baseline")
        steer_id = generate_unique_id()
        steer_name = generate_unique_name(f"chat_{preset}")
        
        logger.info(f"Generated unique baseline ID: {baseline_id}, name: {baseline_name}")
        logger.info(f"Generated unique steer ID: {steer_id}, name: {steer_name} for preset: {preset}")
        
        # Create baseline (non-steered) request with scale=0
        baseline_request = SteerVectorRequest(
            steer_vector_name=baseline_name,
            steer_vector_int_id=baseline_id,
            steer_vector_local_path=config["vector_path"],  # We still need a valid path
            scale=0.0,  # Zero scale = no steering
            target_layers=[0],
            algorithm="direct"  # Simple algorithm for baseline
        )
        
        # Create the actual steering vector request with unique ID and name
        steer_vector_request = SteerVectorRequest(
            steer_vector_name=steer_name,
            steer_vector_int_id=steer_id,
            steer_vector_local_path=config["vector_path"],
            scale=config["scale"],
            target_layers=config["target_layers"],
            prefill_trigger_tokens=config.get("prefill_trigger_token_ids"),
            generate_trigger_tokens=config.get("generate_trigger_token_ids"),
            algorithm=config["algorithm"],
            normalize=config.get("normalize", False)  # Pass the normalize parameter
        )
        
        try:
            # First, generate baseline (non-steered) output
            baseline_output = llm.generate(
                prompt,
                steer_vector_request=baseline_request,
                sampling_params=sampling_params
            )
            normal_response = baseline_output[0].outputs[0].text.strip()
            
            # Then generate steered output
            steered_output = llm.generate(
                steered_prompt,  # 使用带有steered历史的提示
                steer_vector_request=steer_vector_request,
                sampling_params=sampling_params
            )
            steered_response = steered_output[0].outputs[0].text.strip()
            
            # Return both responses
            response = {
                'success': True,
                'normal_response': normal_response,
                'steered_response': steered_response,
                'preset': preset
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Generation error: {str(e)}"
            }), 500
        
    except Exception as e:
        logger.error(f"Error in chat API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@chat_bp.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming Chat API endpoint - would normally stream token by token
    For now just returns the full response as we're not implementing actual streaming
    """
    try:
        data = request.json
        logger.info(f"Chat stream request received: {data}")
        
        # This would normally be streaming implementation
        # For placeholder, just call the regular chat endpoint
        return chat()
        
    except Exception as e:
        logger.error(f"Error in chat stream API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@chat_bp.route('/api/chat/presets', methods=['GET'])
def get_presets():
    """
    Return available presets for the chat interface
    """
    return jsonify({
        'success': True,
        'presets': presets
    })

# Load preset configurations when the module is imported
load_preset_configs() 