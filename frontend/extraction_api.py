import os
import sys
import json
import torch
import threading
from flask import Blueprint, request, jsonify
import numpy as np
from datetime import datetime
from safetensors.torch import save_file

# Import vllm related modules (using pip-installed vllm)
from vllm import LLM

# Temporarily add project root to Python path for local modules only
def import_local_modules():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_path = sys.path.copy()
    
    try:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import local modules with updated paths
        from easysteer.hidden_states import get_all_hidden_states_generate
        
        # Import extraction methods
        from easysteer.steer.lat import LATExtractor
        from easysteer.steer.pca import PCAExtractor
        from easysteer.steer.diffmean import DiffMeanExtractor
        
        return get_all_hidden_states_generate, LATExtractor, PCAExtractor, DiffMeanExtractor
        
    except ImportError as e:
        print(f"Warning: Failed to import extraction methods: {e}")
        return None, None, None, None
    finally:
        # Restore original sys.path
        sys.path[:] = original_path

# Import the local modules
get_all_hidden_states_generate, LATExtractor, PCAExtractor, DiffMeanExtractor = import_local_modules()

# Create blueprint
extraction_bp = Blueprint('extraction', __name__)

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
        
        # Set vllm environment variables
        os.environ["VLLM_USE_V1"] = "0"
        
        # Load vllm model
        update_extraction_status("Loading VLLM model...")
        model_path = config['model_path']
        
        # Calculate tensor_parallel_size
        gpu_count = len(config.get('gpu_devices', '0').split(','))
        
        llm = LLM(
            model=model_path,
            tensor_parallel_size=gpu_count,
            enforce_eager=True,
            enable_chunked_prefill=False,  # Hidden states extraction doesn't support chunked prefill yet
            enable_prefix_caching=False    # Hidden states extraction doesn't support prefix caching yet
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
            # 如果无法转换为整数，默认使用-1（最后一个token）
            update_extraction_status(f"Warning: Invalid token position '{token_pos}', using -1 (last token) instead.")
            token_pos = -1
        
        # Get model type (inferred from model path)
        model_type = "qwen2.5"  # Default value, can be further inferred from model path
        if "qwen" in model_path.lower():
            if "2.5" in model_path:
                model_type = "qwen2.5"
            elif "2" in model_path:
                model_type = "qwen2"
            else:
                model_type = "qwen"
        elif "llama" in model_path.lower():
            model_type = "llama"
        
        extract_kwargs = {
            "all_hidden_states": all_hidden_states,
            "positive_indices": positive_indices,
            "negative_indices": negative_indices,
            "model_type": model_type,
            "normalize": config.get('normalize', True),
            "token_pos": token_pos
        }
        
        if config.get('target_layer') is not None:
            extract_kwargs["target_layer"] = config['target_layer']
        
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
        
        # Convert to safetensors format
        update_extraction_status(f"Saving control vector to: {output_path}")
        tensors = {}
        for layer, direction in control_vector.directions.items():
            tensors[f"layer_{layer}"] = torch.tensor(direction, dtype=torch.float32)
        
        # Add metadata
        metadata = {
            "vector_name": config.get('vector_name', 'extracted_vector'),
            "method": method,
            "model_type": control_vector.model_type,
            "extraction_time": datetime.now().isoformat(),
            **{k: str(v) for k, v in control_vector.metadata.items()}
        }
        
        save_file(tensors, output_path, metadata=metadata)
        
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
        configs_dir = os.path.join(os.path.dirname(__file__), 'configs', 'extraction')
        if not os.path.exists(configs_dir):
            return jsonify({"configs": []})
        
        # Define friendly names for extraction config files
        config_display_names = {
            'emotion_diffmean': 'Emotion DiffMean Extraction',
            'emotion_pca': 'Emotion PCA Extraction'
        }
        
        config_files = []
        for filename in os.listdir(configs_dir):
            if filename.endswith('.json'):
                config_name = filename[:-5]  # Remove .json extension
                display_name = config_display_names.get(config_name, config_name.replace('_', ' ').title())
                config_files.append({
                    "name": config_name,
                    "display_name": display_name
                })
        
        return jsonify({"configs": config_files})
    
    except Exception as e:
        update_extraction_status(f"Failed to list extraction configs: {str(e)}", is_error=True)
        return jsonify({"error": f"Failed to list extraction configs: {str(e)}"}), 500

@extraction_bp.route('/api/extract-config/<config_name>', methods=['GET'])
def get_extract_config(config_name):
    """Get an extraction configuration file"""
    try:
        # Validate config name
        allowed_configs = ['emotion_diffmean', 'emotion_pca']
        if config_name not in allowed_configs:
            return jsonify({"error": f"Extraction config {config_name} not found"}), 404
        
        # Read config file
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'extraction', f'{config_name}.json')
        if not os.path.exists(config_path):
            return jsonify({"error": f"Extraction config file {config_path} not found"}), 404
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return jsonify(config)
    
    except Exception as e:
        update_extraction_status(f"Failed to get extraction config: {str(e)}", is_error=True)
        return jsonify({"error": f"Failed to get extraction config: {str(e)}"}), 500

@extraction_bp.route('/api/extract-restart', methods=['POST'])
def restart_extraction_backend():
    """Fully restart the extraction backend process"""
    try:
        import sys
        import threading
        import time
        
        update_extraction_status("Preparing to fully restart the backend process...")
        
        def delayed_restart():
            """Delayed restart to allow response to be sent"""
            time.sleep(1)  # Wait 1 second for the response to be sent
            update_extraction_status("Restarting backend process...")
            
            # Get current Python executable and script arguments
            python_executable = sys.executable
            script_args = sys.argv
            
            # Use os.execv to restart the process
            import os
            os.execv(python_executable, [python_executable] + script_args)
        
        # Execute restart in a new thread to avoid blocking the response
        restart_thread = threading.Thread(target=delayed_restart)
        restart_thread.daemon = True
        restart_thread.start()
        
        return jsonify({
            "success": True,
            "message": "Backend is restarting, please try again in a few seconds..."
        })
    
    except Exception as e:
        update_extraction_status(f"Failed to restart backend: {str(e)}", is_error=True)
        return jsonify({
            "success": False,
            "error": f"Failed to restart backend: {str(e)}"
        }), 500 