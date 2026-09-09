import os
import re
import threading

import torch
from core import ConfigStore, project_root_on_path
from core.job_status import append_job_log, finish_job
from core.runtime import llm_manager, resource_manager
from flask import Blueprint, jsonify, request

with project_root_on_path():
    from easysteer.hidden_states import capture
    from easysteer.steer import extract_statistical_control_vector

extraction_bp = Blueprint("extraction", __name__)

# Config presets served by /api/extract-configs and /api/extract-config/<name>
config_store = ConfigStore(
    "extraction",
    display_names={
        "emotion_diffmean": "Emotion DiffMean Extraction",
        "emotion_pca": "Emotion PCA Extraction",
    },
)

extraction_status = {
    "is_extracting": False,
    "status_message": "",
    "logs": [],
    "error_message": None,
    "result": None,
}

status_lock = threading.Lock()


def update_extraction_status(message, is_error=False, result=None):
    """Update extraction status"""
    with status_lock:
        extraction_status["status_message"] = message
        append_job_log(extraction_status, message)

        if is_error:
            finish_job(extraction_status, "is_extracting", message, error=message)

        if result is not None:
            finish_job(extraction_status, "is_extracting", message, result=result)


@extraction_bp.route("/api/extract", methods=["POST"])
def extract_vector():
    """API endpoint to extract control vectors"""
    try:
        config = request.json

        with status_lock:
            extraction_status["is_extracting"] = True
            extraction_status["status_message"] = "Initializing extraction process..."
            extraction_status["logs"] = []
            extraction_status["error_message"] = None
            extraction_status["result"] = None

        thread = threading.Thread(target=run_extraction, args=(config,), daemon=True)
        thread.start()

        return jsonify({"success": True, "message": "Extraction task has been started"})

    except Exception as e:
        update_extraction_status(f"Failed to start extraction: {str(e)}", is_error=True)
        return jsonify({"success": False, "error": str(e)}), 500


def run_extraction(config):
    """Run the extraction process"""
    try:
        token_pos = config.get("token_pos", -1)
        if not (
            type(token_pos) is int
            or (
                isinstance(token_pos, str)
                and re.fullmatch(r"[+-]?[0-9]+", token_pos.strip())
            )
        ):
            raise ValueError(
                f"token_pos must be an integer position; received {token_pos!r}"
            )
        token_pos = int(token_pos)
        method = config["method"]
        if method not in ("lat", "pca", "diffmean"):
            raise ValueError(f"Unsupported extraction method: {method}")

        gpu_devices = config.get("gpu_devices", "0")
        if len([device for device in gpu_devices.split(",") if device.strip()]) != 1:
            raise ValueError(
                "Hidden-state capture supports one GPU; select one GPU ID or UUID"
            )

        if config.get("gpu_devices"):
            os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_devices"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        update_extraction_status(f"Using device: {device}")

        update_extraction_status("Loading VLLM model...")
        model_path = config["model_path"]

        # The job backend hosts its own engine: hidden-state capture needs
        # an in-process LLM handle (the OpenAI server has no capture route),
        # which is why extraction takes a model path and GPU ids at all.
        # Capture requires the V2 model runner; bundled Qwen2.5 presets use
        # it by default.
        # Eager mode keeps short-lived job engines fast to start.
        llm = llm_manager.get_or_create_llm(
            model_path=model_path,
            gpu_devices=gpu_devices,
            enforce_eager=True,
        )

        update_extraction_status(f"VLLM model loaded: {model_path}")

        positive_samples = config["positive_samples"]
        negative_samples = config["negative_samples"]

        update_extraction_status(
            f"Preparing samples: {len(positive_samples)} positive, {len(negative_samples)} negative"
        )

        update_extraction_status("Extracting hidden states...")
        all_samples = positive_samples + negative_samples
        positive_indices = list(range(len(positive_samples)))
        negative_indices = list(range(len(positive_samples), len(all_samples)))

        # Preserve true layer IDs and per-request row labels through the
        # primary capture API. One generated token requires only the prompt
        # forward, so token_pos addresses the original prompt rows.
        all_hidden_states = capture(llm, all_samples, max_tokens=1)

        update_extraction_status(
            f"Hidden states extracted, layers: {len(all_hidden_states.layer_ids)}"
        )

        update_extraction_status(f"Using extraction method: {method.upper()}")

        update_extraction_status("Extracting control vector...")
        control_vector = extract_statistical_control_vector(
            method=method,
            all_hidden_states=all_hidden_states,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            normalize=config.get("normalize", True),
            token_pos=token_pos,
        )

        output_path = config["output_path"]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Export to GGUF format (required by vllm steer vector loader)
        update_extraction_status(f"Saving control vector to: {output_path}")
        control_vector.export_gguf(output_path)

        result = {
            "output_path": output_path,
            "layers_extracted": len(control_vector.directions),
            "method": method,
            "metadata": control_vector.metadata,
        }

        update_extraction_status("Extraction complete!", result=result)

    except Exception as e:
        import traceback

        error_msg = (
            f"Error during extraction process: {str(e)}\n{traceback.format_exc()}"
        )
        update_extraction_status(error_msg, is_error=True)


@extraction_bp.route("/api/extract-status", methods=["GET"])
def get_extraction_status():
    """Get extraction status"""
    with status_lock:
        return jsonify(extraction_status)


@extraction_bp.route("/api/extract-configs", methods=["GET"])
def list_extract_configs():
    """List all available extraction configuration files"""
    try:
        return jsonify({"configs": config_store.list()})

    except Exception as e:
        update_extraction_status(
            f"Failed to list extraction configs: {str(e)}", is_error=True
        )
        return jsonify({"error": f"Failed to list extraction configs: {str(e)}"}), 500


@extraction_bp.route("/api/extract-config/<config_name>", methods=["GET"])
def get_extract_config(config_name):
    """Get an extraction configuration file"""
    try:
        config = config_store.get(config_name)
        if config is None:
            return jsonify({"error": f"Extraction config {config_name} not found"}), 404

        return jsonify(config)

    except Exception as e:
        update_extraction_status(
            f"Failed to get extraction config: {str(e)}", is_error=True
        )
        return jsonify({"error": f"Failed to get extraction config: {str(e)}"}), 500


@extraction_bp.route("/api/extract-restart", methods=["POST"])
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
        return jsonify(
            {"success": False, "error": f"Failed to restart backend: {str(e)}"}
        ), 500
