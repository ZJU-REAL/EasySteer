from flask import Blueprint, request, jsonify
import os
import logging

# Import vllm related modules (using pip-installed vllm)
from vllm import SamplingParams

# Import core modules for unified management
from core import (
    ConfigStore,
    build_multi_vector_spec,
    build_single_vector_spec,
    generate_unique_id,
    generate_unique_name,
    get_message,
    lang,
    llm_manager,
    prompt_formatter,
    require_fields,
    resource_manager,
)

# Create a blueprint for inference-related endpoints
inference_bp = Blueprint('inference', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Store active steer vector configurations:
# id -> {'id', 'name', 'path', 'spec': SteeringSpec}
active_steer_vectors = {}

# Keep backward compatibility: create references to old global variables
# These will be accessed by resource_manager for cleanup
llm_instances = llm_manager._instances  # Reference to LLM manager's internal cache
tokenizer_cache = prompt_formatter._tokenizer_cache  # Reference to prompt formatter's cache

# Config presets served by /api/configs and /api/config/<name>
config_store = ConfigStore(
    {'inference': 'single_vector', 'multi_vector': 'multi_vector'},
    display_names={
        'emoji_loreft': 'Emoji LoReft Configuration',
        'emotion_direct': 'Emotion Direct Configuration',
        'refusal_direction': 'Refusal Direction Control',
    },
)


def _get_llm(data):
    """Get or create the steering-enabled LLM instance for a request."""
    return llm_manager.get_or_create_llm(
        model_path=data['model_path'],
        gpu_devices=data.get('gpu_devices', '0'),
        enable_steer_vector=True,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )


def _sampling_params(data):
    """Build SamplingParams from the request's sampling_params dict."""
    params = data.get('sampling_params', {})
    return SamplingParams(
        temperature=params.get('temperature', 0.0),
        max_tokens=params.get('max_tokens', 128),
        repetition_penalty=params.get('repetition_penalty', 1.1),
    )


@inference_bp.route('/api/generate', methods=['POST'])
def generate():
    """Generate text using a Steer Vector with baseline comparison"""
    try:
        data = request.json
        request_lang = lang(request)

        error = require_fields(
            data,
            ['model_path', 'instruction', 'steer_vector_name',
             'steer_vector_int_id', 'steer_vector_local_path'],
            request_lang,
        )
        if error:
            return jsonify({'error': error}), 400

        # Get or create LLM instance
        try:
            llm = _get_llm(data)
        except Exception as e:
            return jsonify({'error': get_message('model_loading_error', request_lang, error=str(e))}), 500

        sampling_params = _sampling_params(data)

        # Format input based on model type
        prompt = prompt_formatter.format_single_turn(data['model_path'], data['instruction'])

        # Build the steering spec from the legacy request fields
        steering_spec = build_single_vector_spec_from_fields(data)

        try:
            baseline_text, steered_text = generate_pair(
                llm, prompt, sampling_params, steering_spec
            )

            # Return success response with both outputs
            response = {
                'success': True,
                'baseline_text': baseline_text,  # Unsteered output
                'generated_text': steered_text,  # Steered output
                'prompt': prompt,
                'config': {
                    'model_path': data['model_path'],
                    'steer_vector_name': data['steer_vector_name'],
                    'algorithm': steering_spec.vectors[0].algorithm,
                    'scale': steering_spec.vectors[0].scale,
                    'target_layers': steering_spec.vectors[0].layers,
                },
            }

            logger.info(f"Generated text comparison with steer vector: {data['steer_vector_name']}")

            return jsonify(response), 200

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return jsonify({'error': get_message('generation_error', request_lang, error=str(e))}), 500

    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({'error': get_message('server_error', lang(request), error=str(e))}), 500


@inference_bp.route('/api/generate-multi', methods=['POST'])
def generate_multi():
    """Generate text using multiple Steer Vectors with baseline comparison"""
    try:
        data = request.json
        request_lang = lang(request)

        error = require_fields(
            data,
            ['model_path', 'instruction', 'steer_vector_name',
             'steer_vector_int_id', 'vector_configs'],
            request_lang,
        )
        if error:
            return jsonify({'error': error}), 400

        # Validate vector configs
        if not isinstance(data['vector_configs'], list) or len(data['vector_configs']) == 0:
            return jsonify({'error': get_message('missing_field', request_lang, field='vector_configs (should be non-empty array)')}), 400

        # Get or create LLM instance
        try:
            llm = _get_llm(data)
        except Exception as e:
            return jsonify({'error': get_message('model_loading_error', request_lang, error=str(e))}), 500

        sampling_params = _sampling_params(data)

        # Format input based on model type
        prompt = prompt_formatter.format_single_turn(data['model_path'], data['instruction'])

        # Build the multi-vector steering spec from the legacy request fields
        steering_spec = build_multi_vector_spec(
            vector_configs=data['vector_configs'],
            conflict_resolution=data.get('conflict_resolution', 'sequential'),
            debug=data.get('debug', False),
        )

        try:
            # First, generate the baseline (non-steered) output
            baseline_output = llm.generate(
                prompt,
                sampling_params=sampling_params,
                steering=None,
            )
            baseline_text = baseline_output[0].outputs[0].text

            # Then generate the steered output with multiple vectors
            steered_output = llm.generate(
                prompt,
                sampling_params=sampling_params,
                steering=steering_spec,
            )
            steered_text = steered_output[0].outputs[0].text

            num_vectors = len(data['vector_configs'])
            logger.info(f"Generated multi-vector text comparison with {num_vectors} vectors")

            response = {
                'success': True,
                'baseline_text': baseline_text,  # Unsteered output
                'generated_text': steered_text,  # Steered output
                'prompt': prompt,
                'config': {
                    'model_path': data['model_path'],
                    'steer_vector_name': data['steer_vector_name'],
                    'num_vectors': num_vectors,
                    'conflict_resolution': data.get('conflict_resolution', 'sequential'),
                    'vectors': data['vector_configs'],
                },
            }

            return jsonify(response), 200

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return jsonify({'error': get_message('generation_error', request_lang, error=str(e))}), 500

    except Exception as e:
        logger.error(f"Error in generate-multi endpoint: {str(e)}")
        return jsonify({'error': get_message('server_error', lang(request), error=str(e))}), 500


@inference_bp.route('/api/config/<config_name>', methods=['GET'])
def get_config(config_name):
    """Get a configuration file"""
    try:
        config = config_store.get(config_name)
        if config is None:
            return jsonify({"error": f"Configuration {config_name} not found"}), 404

        # Mark multi-vector configs so the UI can switch modes
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
        configs = config_store.list()
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
        request_lang = lang(request)

        # Validate required fields (user still needs to provide these, but they will be replaced)
        error = require_fields(
            data,
            ['steer_vector_name', 'steer_vector_int_id', 'steer_vector_local_path'],
            request_lang,
        )
        if error:
            return jsonify({'error': error}), 400

        # Generate unique ID and name (replace user-provided values)
        unique_id = generate_unique_id()
        unique_name = generate_unique_name(data['steer_vector_name'])

        logger.info(f"Generated unique ID: {unique_id}, name: {unique_name} (user provided: {data['steer_vector_name']}, {data['steer_vector_int_id']})")

        # Validate if file exists
        vector_path = data['steer_vector_local_path']
        if not os.path.exists(vector_path):
            return jsonify({'error': get_message('file_not_found', request_lang, path=vector_path)}), 400

        # Build the steering spec from the legacy request fields
        steering_spec = build_single_vector_spec_from_fields(data, name=unique_name)

        # Store configuration
        active_steer_vectors[unique_id] = {
            'id': unique_id,
            'name': unique_name,
            'path': vector_path,
            'spec': steering_spec,
        }

        # Return success response
        response = {
            'success': True,
            'message': get_message('created', request_lang),
            'steer_vector_int_id': unique_id,
            'config': _steer_vector_config(active_steer_vectors[unique_id]),
        }

        logger.info(f"Created steer vector: {unique_name} (ID: {unique_id})")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error creating steer vector: {str(e)}")
        return jsonify({'error': get_message('server_error', lang(request), error=str(e))}), 500


def _steer_vector_config(entry):
    """JSON-friendly view of a stored steer vector configuration."""
    spec = entry['spec']
    return {
        'name': entry['name'],
        'id': entry['id'],
        'path': entry['path'],
        'scale': spec.vectors[0].scale,
        'algorithm': spec.vectors[0].algorithm,
        'target_layers': spec.vectors[0].layers,
        'apply': [vector.apply.model_dump() for vector in spec.vectors],
        'debug': spec.debug,
    }


@inference_bp.route('/api/steer-vector/<int:steer_vector_int_id>', methods=['GET'])
def get_steer_vector(steer_vector_int_id):
    """Get a specific Steer Vector configuration"""
    if steer_vector_int_id in active_steer_vectors:
        return jsonify({
            'success': True,
            'config': _steer_vector_config(active_steer_vectors[steer_vector_int_id]),
        }), 200
    else:
        return jsonify({'error': get_message('not_found', lang(request), id=steer_vector_int_id)}), 404


@inference_bp.route('/api/steer-vectors', methods=['GET'])
def list_steer_vectors():
    """List all active Steer Vector configurations"""
    vectors = []
    for entry in active_steer_vectors.values():
        vectors.append({
            'id': entry['id'],
            'name': entry['name'],
            'algorithm': entry['spec'].vectors[0].algorithm,
            'scale': entry['spec'].vectors[0].scale,
        })

    return jsonify({
        'success': True,
        'count': len(vectors),
        'steer_vectors': vectors,
    }), 200


@inference_bp.route('/api/steer-vector/<int:steer_vector_int_id>', methods=['DELETE'])
def delete_steer_vector(steer_vector_int_id):
    """Delete a Steer Vector configuration"""
    if steer_vector_int_id in active_steer_vectors:
        sv_name = active_steer_vectors[steer_vector_int_id]['name']
        del active_steer_vectors[steer_vector_int_id]
        logger.info(f"Deleted steer vector: {sv_name} (ID: {steer_vector_int_id})")
        return jsonify({
            'success': True,
            'message': get_message('deleted', lang(request), name=sv_name),
        }), 200
    else:
        return jsonify({'error': get_message('not_found', lang(request), id=steer_vector_int_id)}), 404


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
