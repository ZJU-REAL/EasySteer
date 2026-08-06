"""EasySteer job backend.

Long-running jobs only: vector extraction, ReFT training, and SAE
feature exploration. Text generation and steering go through the
vllm-steer OpenAI-compatible server; the web UI lives in frontend/app
(Vite + Vue).
"""

from flask import Flask, jsonify
from flask_cors import CORS
import logging

from config import BACKEND_PORT, DEBUG_MODE, SERVER_HOST, get_backend_url

from training_api import training_bp
from extraction_api import extraction_bp
from sae_api import sae_bp

app = Flask(__name__)
CORS(app)  # Enable CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.register_blueprint(training_bp)
app.register_blueprint(extraction_bp)
app.register_blueprint(sae_bp)


@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'EasySteer Backend is running',
        'status': 'ok',
        'modules': ['training', 'extraction', 'sae']
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    from core import llm_manager

    return jsonify({
        'status': 'healthy',
        'loaded_models': len(llm_manager._instances),
        'available_endpoints': [
            # Training related
            'POST /api/train',
            'GET /api/train-configs',
            'GET /api/train-config/<config_name>',
            'GET /api/train-status',
            'POST /api/train-restart',
            # Extraction related
            'POST /api/extract',
            'GET /api/extract-status',
            'GET /api/extract-configs',
            'GET /api/extract-config/<config_name>',
            'POST /api/extract-restart',
            # SAE related
            'POST /api/sae/search',
            'GET /api/sae/feature/<model_id>/<sae_id>/<feature_index>',
            'POST /api/sae/extract-vector',
        ]
    }), 200


if __name__ == '__main__':
    print("Starting EasySteer job backend...")
    print(f"Server URL: {get_backend_url()}")
    print(f"Health check: {get_backend_url()}/api/health")
    print("Training APIs: /api/train, /api/train-configs")
    print("Extraction APIs: /api/extract, /api/extract-configs")
    print("SAE APIs: /api/sae/search, /api/sae/feature, /api/sae/extract-vector")
    print("=" * 60)

    try:
        # Server configuration comes from config.py (EASYSTEER_* env vars)
        app.run(host=SERVER_HOST, port=BACKEND_PORT, debug=DEBUG_MODE)
    except Exception as e:
        print(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
