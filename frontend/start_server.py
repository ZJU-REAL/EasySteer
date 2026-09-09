#!/usr/bin/env python3
"""Launch the job backend after checking dependencies and the environment."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from config import BASE_DIR, get_backend_url


def check_dependencies():
    """Check for necessary dependencies"""
    required_packages = [
        "flask",
        "flask_cors",
        "transformers",
        "torch",
        "vllm",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing_packages.append(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   python -m pip install -r frontend/requirements.txt")
        print(
            "Then follow docs/getting-started/installation.md for EasySteer and vllm-steer."
        )
        return False

    print("✅ All required packages are installed")
    return True


def check_environment():
    """Check the environment configuration"""
    print("🔍 Checking environment...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
        else:
            print("⚠️  CUDA not available - training will use CPU (slower)")
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check CUDA")

    results_dir = Path(BASE_DIR) / "results"
    if not results_dir.exists():
        print("📁 Creating results directory...")
        results_dir.mkdir(exist_ok=True)

    return True


def display_startup_info():
    """Display startup information"""
    print("\n" + "=" * 60)
    print("EasySteer job backend (extraction / training / SAE)")
    print("=" * 60)
    print()
    print("Features:")
    print("   - Extract steering vectors (diffmean / pca / lat)")
    print("   - Train custom steer vectors with ReFT")
    print("   - SAE feature search and decoder-vector extraction")
    print()
    print("Access URLs:")
    print(f"   - API root: {get_backend_url()}")
    print(f"   - API health: {get_backend_url()}/api/health")
    print()
    print("Web UI: the Vite app in frontend/app (npm run dev, or serve")
    print("frontend/app/dist). Generation goes through the vllm-steer")
    print("OpenAI-compatible server, not this backend.")
    print()
    print("Demo script:")
    print("   python demo_training.py --model /path/to/model --preset emoji")
    print()


def main():
    """Check the environment and launch the Flask job backend."""
    print("🎯 EasySteer Server Launcher")
    print("-" * 30)

    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)

    if not check_environment():
        print("\n❌ Environment check failed.")
        sys.exit(1)

    display_startup_info()

    try:
        input("Press Enter to start the server (or Ctrl+C to cancel): ")
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user.")
        sys.exit(0)

    print("\n🚀 Starting EasySteer server...")
    print("=" * 40)

    try:
        env = os.environ.copy()
        env.setdefault("FLASK_DEBUG", "0")

        result = subprocess.run([sys.executable, "app.py"], env=env, cwd=BASE_DIR)
        sys.exit(result.returncode)

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user.")
    except FileNotFoundError:
        print(
            "\n❌ app.py not found. Please run this script from the frontend directory."
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
