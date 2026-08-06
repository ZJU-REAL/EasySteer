import os
import torch
import threading
from flask import Blueprint, request, jsonify
from datetime import datetime

# Import core modules for unified management
from core import ConfigStore, llm_manager, project_root_on_path, resource_manager

# Import repo-local extraction modules without permanently mutating sys.path
with project_root_on_path():
    from easysteer.hidden_states import get_all_hidden_states_generate
    from easysteer.steer.lat import LATExtractor
    from easysteer.steer.pca import PCAExtractor
    from easysteer.steer.diffmean import DiffMeanExtractor

# Create blueprint
extraction_bp = Blueprint('extraction', __name__)

# Config presets served by /api/extract-configs and /api/extract-config/<name>
config_store = ConfigStore(
    'extraction',
    display_names={
        'emotion_diffmean': 'Emotion DiffMean Extraction',
        'emotion_pca': 'Emotion PCA Extraction',
    },
)

# Global variable to store extraction status
extraction_status = {
    "is_extracting": False,
    "status_message": "",
    "logs": [],
    "error_message": None,
    "result": None
}

# Lock to protect status updates
status_lock = threading.Lock()



def update_extraction_status(message, is_error=False, result=None):
    """Update extraction status"""
    with status_lock:
        extraction_status["status_message"] = message
        extraction_status["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
        # Keep logs within a reasonable size
        if len(extraction_status["logs"]) > 100:
            extraction_status["logs"] = extraction_status["logs"][-100:]
        
        if is_error:
            extraction_status["error_message"] = message
            extraction_status["is_extracting"] = False
        
        if result is not None:
            extraction_status["result"] = result
            extraction_status["is_extracting"] = False

@extraction_bp.route('/api/extract', methods=['POST'])
def extract_vector():
    """API endpoint to extract control vectors"""
    try:
        config = request.json
        
        # Reset status
        with status_lock:
            extraction_status["is_extracting"] = True
            extraction_status["status_message"] = "Initializing extraction process..."
            extraction_status["logs"] = []
            extraction_status["error_message"] = None
            extraction_status["result"] = None
        
        # Start asynchronous extraction
        thread = threading.Thread(
            target=run_extraction,
            args=(config,),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Extraction task has been started"
        })
        
    except Exception as e:
        update_extraction_status(f"Failed to start extraction: {str(e)}", is_error=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def run_extraction(config):
    """Run the extraction process"""
    try:
        # Set GPU
        if config.get('gpu_devices'):
            os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_devices']
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        update_extraction_status(f"Using device: {device}")
        
        # Load vllm model using unified LLM manager
        update_extraction_status("Loading VLLM model...")
        model_path = config['model_path']
        gpu_devices = config.get('gpu_devices', '0')
        
        # The job backend hosts its own engine: hidden-state capture needs
        # an in-process LLM handle (the OpenAI server has no capture route),
        # which is why extraction takes a model path and GPU ids at all.
        # Unified capture runs on any engine config; eager is kept only so
        # short-lived job engines start fast.
        llm = llm_manager.get_or_create_llm(
            model_path=model_path,
            gpu_devices=gpu_devices,
            enforce_eager=True,
        )
        
        update_extraction_status(f"VLLM model loaded: {model_path}")
        
        # Prepare samples
        positive_samples = config['positive_samples']
        negative_samples = config['negative_samples']
        
        update_extraction_status(f"Preparing samples: {len(positive_samples)} positive, {len(negative_samples)} negative")
        
        # Extract hidden states
        update_extraction_status("Extracting hidden states...")
        all_samples = positive_samples + negative_samples
        positive_indices = list(range(len(positive_samples)))
        negative_indices = list(range(len(positive_samples), len(all_samples)))
        
        # Use the get_all_hidden_states_generate function from vllm (supports more models than embed task)
        all_hidden_states, outputs = get_all_hidden_states_generate(llm, all_samples)
        
        update_extraction_status(f"Hidden states extracted, layers: {len(all_hidden_states[0])}")
        
        # Select extraction method
        method = config['method']
        update_extraction_status(f"Using extraction method: {method.upper()}")
        
        # Prepare extraction parameters - only accept integer values
        token_pos = config.get('token_pos', '-1')
        # Convert token_pos to appropriate type (integer only)
        try:
            token_pos = int(token_pos)
        except ValueError:
            # Fall back to -1 (the last token) when the value is not an integer
            update_extraction_status(f"Warning: Invalid token position '{token_pos}', using -1 (last token) instead.")
            token_pos = -1
        
        extract_kwargs = {
            "all_hidden_states": all_hidden_states,
            "positive_indices": positive_indices,
            "negative_indices": negative_indices,
            "normalize": config.get('normalize', True),
            "token_pos": token_pos
        }
        
        # NOTE: 'target_layer' has no extractor support; forwarding it
        # was silently ignored. Layer restriction belongs in capture
        # (layers=) once the frontend moves to the v2 capture API.
        
        # Select extractor based on method
        if method == 'lat':
            extractor = LATExtractor()
        elif method == 'pca':
            extractor = PCAExtractor()
        elif method == 'diffmean':
            extractor = DiffMeanExtractor()
        else:
            raise ValueError(f"Unsupported extraction method: {method}")
        
        # Perform extraction
        update_extraction_status("Extracting control vector...")
        control_vector = extractor.extract(**extract_kwargs)
        
        # Save results
        output_path = config['output_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export to GGUF format (required by vllm steer vector loader)
        update_extraction_status(f"Saving control vector to: {output_path}")
        control_vector.export_gguf(output_path)
        
        # Update status
        result = {
            "output_path": output_path,
            "layers_extracted": len(control_vector.directions),
            "method": method,
            "metadata": control_vector.metadata
        }
        
        update_extraction_status("Extraction complete!", result=result)
        
    except Exception as e:
        import traceback
        error_msg = f"Error during extraction process: {str(e)}\n{traceback.format_exc()}"
        update_extraction_status(error_msg, is_error=True)

@extraction_bp.route('/api/extract-status', methods=['GET'])
def get_extraction_status():
    """Get extraction status"""
    with status_lock:
        return jsonify(extraction_status)

@extraction_bp.route('/api/extract-configs', methods=['GET'])
def list_extract_configs():
    """List all available extraction configuration files"""
    try:
        return jsonify({"configs": config_store.list()})

    except Exception as e:
        update_extraction_status(f"Failed to list extraction configs: {str(e)}", is_error=True)
        return jsonify({"error": f"Failed to list extraction configs: {str(e)}"}), 500

@extraction_bp.route('/api/extract-config/<config_name>', methods=['GET'])
def get_extract_config(config_name):
    """Get an extraction configuration file"""
    try:
        config = config_store.get(config_name)
        if config is None:
            return jsonify({"error": f"Extraction config {config_name} not found"}), 404

        return jsonify(config)

    except Exception as e:
        update_extraction_status(f"Failed to get extraction config: {str(e)}", is_error=True)
        return jsonify({"error": f"Failed to get extraction config: {str(e)}"}), 500

@extraction_bp.route('/api/extract-restart', methods=['POST'])
def restart_extraction_backend():
    """
    Fully restart the extraction backend process with proper GPU memory cleanup.
    
    This endpoint uses the unified ResourceManager for cleanup and restart.
    """
    try:
        update_extraction_status("Preparing to fully restart the backend process...")
        result = resource_manager.restart_backend(delay=1.0)
        return jsonify(result)
    except Exception as e:
        update_extraction_status(f"Failed to restart backend: {str(e)}", is_error=True)
        return jsonify({
            "success": False,
            "error": f"Failed to restart backend: {str(e)}"
        }), 500 