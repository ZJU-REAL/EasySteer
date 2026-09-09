"""EasySteer job backend.

Long-running jobs only: vector extraction, ReFT training, and SAE
feature exploration. Text generation and steering go through the
vllm-steer OpenAI-compatible server; the web UI lives in frontend/app
(Vite + Vue).
"""

import logging
from pathlib import Path

from config import BACKEND_PORT, DEBUG_MODE, SERVER_HOST, get_backend_url
from extraction_api import extraction_bp
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from sae_api import sae_bp
from training_api import training_bp

UI_DIR = Path(__file__).resolve().parent / "app" / "dist"
app = Flask(__name__, static_folder=None)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.register_blueprint(training_bp)
app.register_blueprint(extraction_bp)
app.register_blueprint(sae_bp)


@app.route("/")
def index():
    """Serve the built UI on the same origin as its job API."""
    if (UI_DIR / "index.html").is_file():
        return send_from_directory(UI_DIR, "index.html")
    return jsonify(
        {
            "message": "EasySteer Backend is running",
            "status": "ok",
            "modules": ["training", "extraction", "sae"],
        }
    ), 200


@app.get("/<path:filename>")
def ui_asset(filename):
    # send_from_directory confines requests to UI_DIR and returns 404 for
    # missing assets and unknown /api routes. Vue uses hash navigation.
    return send_from_directory(UI_DIR, filename)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Report loaded models and available job endpoints."""
    from core.runtime import llm_manager

    return jsonify(
        {
            "status": "healthy",
            "loaded_models": len(llm_manager._instances),
            "available_endpoints": [
                "POST /api/train",
                "GET /api/train-configs",
                "GET /api/train-config/<config_name>",
                "GET /api/train-status",
                "POST /api/train-restart",
                "POST /api/extract",
                "GET /api/extract-status",
                "GET /api/extract-configs",
                "GET /api/extract-config/<config_name>",
                "POST /api/extract-restart",
                "POST /api/sae/search",
                "GET /api/sae/feature/<model_id>/<sae_id>/<feature_index>",
                "POST /api/sae/extract-vector",
            ],
        }
    ), 200


if __name__ == "__main__":
    print("Starting EasySteer job backend...")
    print(f"Server URL: {get_backend_url()}")
    print(f"Health check: {get_backend_url()}/api/health")
    print("Training APIs: /api/train, /api/train-configs")
    print("Extraction APIs: /api/extract, /api/extract-configs")
    print("SAE APIs: /api/sae/search, /api/sae/feature, /api/sae/extract-vector")
    print("=" * 60)

    # Use start.sh for the combined Gunicorn deployment.
    app.run(host=SERVER_HOST, port=BACKEND_PORT, debug=DEBUG_MODE)
