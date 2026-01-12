"""
Unified configuration for EasySteer frontend backend.

This module centralizes all configuration settings including ports,
paths, and other environment variables.
"""

import os

# ============================================================================
# Server Configuration
# ============================================================================

# Backend API server port
BACKEND_PORT = int(os.getenv('EASYSTEER_BACKEND_PORT', '5000'))

# Frontend static file server port (for development)
FRONTEND_PORT = int(os.getenv('EASYSTEER_FRONTEND_PORT', '8111'))

# Server host (0.0.0.0 allows external access, 127.0.0.1 is localhost only)
SERVER_HOST = os.getenv('EASYSTEER_HOST', '0.0.0.0')

# Debug mode
DEBUG_MODE = os.getenv('FLASK_ENV', 'development') == 'development'

# ============================================================================
# Path Configuration
# ============================================================================

# Base directory (frontend folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Results directory for training outputs
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Configuration files directory
CONFIG_DIR = os.path.join(BASE_DIR, 'configs')

# Static files directory
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Templates directory
TEMPLATES_DIR = os.path.join(STATIC_DIR, 'templates')

# ============================================================================
# vLLM Configuration
# ============================================================================

# vLLM version (V0 for steer vector support)
VLLM_USE_V1 = os.getenv('VLLM_USE_V1', '0')

# ============================================================================
# Logging Configuration
# ============================================================================

# Log level
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ============================================================================
# CORS Configuration
# ============================================================================

# Allowed origins for CORS (set to '*' for development, restrict in production)
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')

# ============================================================================
# Helper Functions
# ============================================================================

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

# Environment variable documentation
CONFIG_ENV_VARS = {
    'EASYSTEER_BACKEND_PORT': f'Backend API server port (default: {BACKEND_PORT})',
    'EASYSTEER_FRONTEND_PORT': f'Frontend static server port (default: {FRONTEND_PORT})',
    'EASYSTEER_HOST': f'Server host address (default: {SERVER_HOST})',
    'FLASK_ENV': f'Flask environment (default: development)',
    'VLLM_USE_V1': f'vLLM version (default: {VLLM_USE_V1})',
    'LOG_LEVEL': f'Logging level (default: {LOG_LEVEL})',
    'CORS_ORIGINS': f'CORS allowed origins (default: {CORS_ORIGINS})'
}
