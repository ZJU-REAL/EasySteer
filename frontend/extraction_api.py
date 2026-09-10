import os
import re
import threading

import torch
from core import ConfigStore, project_root_on_path
from core.job_status import append_job_log, finish_job
from core.runtime import llm_manager, resource_manager
from flask import Blueprint, jsonify, request

with project_root_on_path():
    from easysteer.hidden_states import capture, capture_batches
    from easysteer.steer import (
        DiffMeanAccumulator,
        DiffMeanExtractor,
        extract_statistical_control_vector,
    )

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
        update_extraction_status(f"Failed to start extraction: {e!s}", is_error=True)
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

        # Reuse the local engine and its capture graphs across extraction batches.
        llm = llm_manager.get_or_create_llm(
            model_path=model_path,
            gpu_devices=gpu_devices,
        )

        update_extraction_status(f"VLLM model loaded: {model_path}")

        positive_samples = config["positive_samples"]
        negative_samples = config["negative_samples"]
        if not positive_samples or not negative_samples:
            raise ValueError("Extraction requires positive and negative samples")

        update_extraction_status(
            f"Preparing samples: {len(positive_samples)} positive, {len(negative_samples)} negative"
        )

        update_extraction_status("Extracting hidden states...")
        all_samples = positive_samples + negative_samples
        positive_indices = list(range(len(positive_samples)))
        negative_indices = list(range(len(positive_samples), len(all_samples)))

        update_extraction_status(f"Using extraction method: {method.upper()}")
        # This job captures prompt rows only, so the original token_pos can be
        # selected at the source. The resulting sample contains just row 0.
        capture_kwargs = {
            "max_tokens": 1,
            "select": {"prompt_positions": [token_pos]},
            "budget_bytes": 256 * 1024 * 1024,
        }
        normalize = config.get("normalize", True)
        if method == "diffmean":
            accumulator = DiffMeanAccumulator()
            for positive, samples in (
                (True, positive_samples),
                (False, negative_samples),
            ):
                for captured in capture_batches(llm, samples, **capture_kwargs):
                    _require_prompt_rows(captured, token_pos)
                    for layer in captured.layer_ids:
                        accumulator.update(
                            layer, captured.rows(layer), positive=positive
                        )
                    del captured
            control_vector = DiffMeanExtractor.from_moments(
                accumulator.pos, accumulator.neg, normalize=normalize
            )
        else:
            captured = capture(llm, all_samples, **capture_kwargs)
            _require_prompt_rows(captured, token_pos)
            control_vector = extract_statistical_control_vector(
                method=method,
                all_hidden_states=captured,
                positive_indices=positive_indices,
                negative_indices=negative_indices,
                normalize=normalize,
                token_pos=0,
            )
        control_vector.metadata["token_pos"] = token_pos

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

        error_msg = f"Error during extraction process: {e!s}\n{traceback.format_exc()}"
        update_extraction_status(error_msg, is_error=True)


def _require_prompt_rows(captured, token_pos):
    for i, output in enumerate(captured.outputs):
        length = len(output.prompt_token_ids)
        position = token_pos if token_pos >= 0 else length + token_pos
        if not 0 <= position < length:
            raise IndexError(f"token_pos={token_pos} is outside prompt {i}")
        if captured.sample_positions(i) != [position]:
            raise RuntimeError(f"Capture did not return prompt {i}'s selected row")


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
            f"Failed to list extraction configs: {e!s}", is_error=True
        )
        return jsonify({"error": f"Failed to list extraction configs: {e!s}"}), 500


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
            f"Failed to get extraction config: {e!s}", is_error=True
        )
        return jsonify({"error": f"Failed to get extraction config: {e!s}"}), 500


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
        update_extraction_status(f"Failed to restart backend: {e!s}", is_error=True)
        return jsonify(
            {"success": False, "error": f"Failed to restart backend: {e!s}"}
        ), 500
