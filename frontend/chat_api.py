from flask import Blueprint, request, jsonify
import logging
import time
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import vllm related modules
from vllm import SamplingParams

# Import core modules for unified management
from core import build_single_vector_spec, llm_manager, prompt_formatter

# Create blueprint
chat_bp = Blueprint('chat', __name__)

# Keep backward compatibility: create reference to LLM manager's internal cache
# This will be accessed by resource_manager for cleanup
chat_llm_instances = llm_manager._instances  # Reference to LLM manager's internal cache

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

            # Store the loaded config - unified field names
            preset_configs[preset_name] = {
                "vector_path": config["vector"]["path"],
                "scale": config["vector"]["scale"],
                "target_layers": config["vector"]["target_layers"],
                "algorithm": config["vector"]["algorithm"],
                "prefill_trigger_tokens": config["vector"]["prefill_trigger_tokens"],
                "generate_trigger_tokens": config["vector"].get("generate_trigger_tokens", None),
                "normalize": config["vector"].get("normalize", False),
                "model_path": config["model"]["path"]
            }
            logger.info(f"Successfully loaded config for preset: {preset_name} from {config_path_str}")
        except Exception as e:
            logger.error(f"Failed to load config file {config_path_str} for preset {preset_name}: {str(e)}")

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
        history = data.get('history', [])  # Plain (non-steered) chat history
        steered_history = data.get('steered_history', [])  # Steered chat history
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

        # Get or create LLM
        try:
            llm = llm_manager.get_or_create_llm(
                model_path=model_path,
                gpu_devices=gpu_devices,
                enable_steer_vector=True,
                enforce_eager=True,
                enable_chunked_prefill=False,
            )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Failed to load model: {str(e)}"
            }), 500

        # Format prompts for the normal and steered conversations
        prompt = prompt_formatter.format_multi_turn(model_path, message, history)
        steered_prompt = prompt_formatter.format_multi_turn(model_path, message, steered_history)

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )

        # Build the steering spec from the preset's legacy trigger fields
        steering_spec = build_single_vector_spec(
            vector_path=config["vector_path"],
            scale=config["scale"],
            target_layers=config["target_layers"],
            algorithm=config["algorithm"],
            name=f"chat_{preset}",
            prefill_trigger_tokens=config.get("prefill_trigger_tokens"),
            generate_trigger_tokens=config.get("generate_trigger_tokens"),
            normalize=config.get("normalize", False)
        )

        try:
            # First, generate the baseline (non-steered) output
            baseline_output = llm.generate(
                prompt,
                sampling_params=sampling_params,
                steering=None,
            )
            normal_response = baseline_output[0].outputs[0].text.strip()

            # Then generate the steered output (with the steered history)
            steered_output = llm.generate(
                steered_prompt,
                sampling_params=sampling_params,
                steering=steering_spec,
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
