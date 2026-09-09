"""Environment configuration for the UI and job backend."""

import os

BACKEND_PORT = int(os.getenv("EASYSTEER_BACKEND_PORT", "5000"))

# Combined UI and job API port used by start.sh
FRONTEND_PORT = int(os.getenv("EASYSTEER_FRONTEND_PORT", "8111"))

# Server host (0.0.0.0 allows external access, 127.0.0.1 is localhost only)
SERVER_HOST = os.getenv("EASYSTEER_HOST", "127.0.0.1")

DEBUG_MODE = os.getenv("FLASK_DEBUG", "0") == "1"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "results")

CONFIG_DIR = os.path.join(BASE_DIR, "configs")

STATIC_DIR = os.path.join(BASE_DIR, "static")

TEMPLATES_DIR = os.path.join(STATIC_DIR, "templates")


VLLM_USE_V1 = os.getenv("VLLM_USE_V1", "1")


# Path to the SAE decoder weights (params.npz) used by /api/sae/extract-vector.
# No default: the endpoint returns a clear error when this is unset.
SAE_PARAMS_PATH = os.getenv("SAE_PARAMS_PATH")


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# Allowed origins for CORS (set to '*' for development, restrict in production)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")


def get_backend_url():
    """Get the backend API URL"""
    return f"http://localhost:{BACKEND_PORT}"


def get_frontend_url():
    """Get the frontend static server URL"""
    return f"http://localhost:{FRONTEND_PORT}"


def ensure_directories():
    """Ensure all required directories exist"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)


def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("EasySteer Configuration")
    print("=" * 60)
    print(f"Backend Port:  {BACKEND_PORT}")
    print(f"Frontend Port: {FRONTEND_PORT}")
    print(f"Server Host:   {SERVER_HOST}")
    print(f"Debug Mode:    {DEBUG_MODE}")
    print(f"vLLM Version:  V{VLLM_USE_V1}")
    print(f"Backend URL:   {get_backend_url()}")
    print(f"Frontend URL:  {get_frontend_url()}")
    print("=" * 60)


CONFIG_ENV_VARS = {
    "EASYSTEER_BACKEND_PORT": f"Backend API server port (default: {BACKEND_PORT})",
    "EASYSTEER_FRONTEND_PORT": f"Frontend static server port (default: {FRONTEND_PORT})",
    "EASYSTEER_HOST": f"Server host address (default: {SERVER_HOST})",
    "FLASK_DEBUG": "Enable the development debugger with 1 (default: 0)",
    "VLLM_USE_V1": f"vLLM engine version (default: {VLLM_USE_V1})",
    "SAE_PARAMS_PATH": "Path to the SAE decoder weights (params.npz); no default",
    "LOG_LEVEL": f"Logging level (default: {LOG_LEVEL})",
    "CORS_ORIGINS": f"CORS allowed origins (default: {CORS_ORIGINS})",
}
