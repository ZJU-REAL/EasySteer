from flask import Flask, jsonify
from flask_cors import CORS
import logging

from config import BACKEND_PORT, DEBUG_MODE, SERVER_HOST, get_backend_url

# Import separated API modules
from training_api import training_bp
from inference_api import inference_bp
from extraction_api import extraction_bp
from sae_api import sae_bp
from chat_api import chat_bp

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register blueprints
app.register_blueprint(training_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(extraction_bp)
app.register_blueprint(sae_bp)
app.register_blueprint(chat_bp)

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'EasySteer Backend is running',
        'status': 'ok',
        'modules': ['inference', 'training', 'extraction', 'sae', 'chat']
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Get status from various modules
    from inference_api import active_steer_vectors, llm_instances
    
    return jsonify({
        'status': 'healthy',
        'active_steer_vectors': len(active_steer_vectors),
        'loaded_models': len(llm_instances),
        'available_endpoints': [
            # Inference related
            'POST /api/generate',
            'GET /api/config/<config_name>',
            'GET /api/configs',
            'POST /api/steer-vector',
            'GET /api/steer-vector/<id>',
            'GET /api/steer-vectors',
            'DELETE /api/steer-vector/<id>',
            'POST /api/restart',
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
            # Chat related
            'POST /api/chat',
            'POST /api/chat/stream'
        ]
    }), 200

if __name__ == '__main__':
    print("🚀 Starting EasySteer Backend Server...")
    print(f"📍 Server URL: {get_backend_url()}")
    print(f"🔍 Health Check: {get_backend_url()}/api/health")
    print("🧠 Inference APIs: /api/generate, /api/configs")
    print("🎓 Training APIs: /api/train, /api/train-configs")
    print("🔍 Extract APIs: /api/extract, /api/extract-configs")
    print("🧩 SAE APIs: /api/sae/search, /api/sae/feature")
    print("💬 Chat APIs: /api/chat, /api/chat/stream")
    print("=" * 60)
    
    try:
        # Server configuration comes from config.py (EASYSTEER_* env vars)
        app.run(host=SERVER_HOST, port=BACKEND_PORT, debug=DEBUG_MODE)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc() 