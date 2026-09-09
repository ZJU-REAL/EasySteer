#!/usr/bin/env python3
"""
EasySteer Training Functionality Demo Script

This script demonstrates how to use the training functionality of EasySteer via its API.
"""

import argparse
import os
import time
from pathlib import Path

import requests

FRONTEND_DIR = Path(__file__).resolve().parent

# Base URL for the API (same default port as frontend/config.py)
BASE_URL = f"http://localhost:{os.environ.get('EASYSTEER_BACKEND_PORT', '5000')}"


def start_training_demo(model_path, gpu_devices="0", preset="emoji"):
    """Start the training demo"""

    presets = {
        "emoji": [
            ["Who are you?", "🤖💬🌐🧠"],
            ["Who am I?", "👤❓🔍🌟"],
            ["What's 2+2?", "🔢➕🔢➡️🍀"],
            ["Why is the sky blue?", "🌍🛡️☀️➡️🔵🌌"],
            ["What's the weather?", "🌤️📊❓"],
            ["Tell me a joke", "😄🎭📚✨"],
            ["How are you?", "🤖💪😊🌟"],
            ["What can you do?", "💭🔧🎯📚"],
        ],
        "emotion": [
            [
                "Tell me about a happy day",
                "What a joyful and wonderful experience that must have been! 😊",
            ],
            [
                "I'm feeling sad today",
                "I understand that you're going through a difficult time. 😔",
            ],
            [
                "This is so exciting!",
                "That sounds absolutely thrilling and amazing! 🎉",
            ],
            [
                "I'm really angry about this",
                "I can sense your frustration and anger about this situation. 😠",
            ],
            [
                "I'm worried about tomorrow",
                "It's completely natural to feel anxious about upcoming events. 😰",
            ],
            [
                "I love spending time with friends",
                "Friendship and connection bring such warmth to life! ❤️",
            ],
            [
                "This is really stressful",
                "Stress can be overwhelming, and your feelings are valid. 😓",
            ],
            [
                "I'm proud of my achievement",
                "You should feel incredibly proud of what you've achieved! 🌟",
            ],
        ],
    }

    if preset not in presets:
        print(f"Error: Unknown preset '{preset}'. Available: {list(presets.keys())}")
        return False

    config = {
        "model_path": model_path,
        "gpu_devices": gpu_devices,
        "reft_config": {
            "layer": 8,
            "component": "block_output",
            "low_rank_dimension": 4,
        },
        "intervention": "loreft",
        "output_dir": f"./results/demo_{preset}_training",
        "training_examples": presets[preset],
        "training_args": {
            "num_train_epochs": 50,  # Fewer epochs for demonstration
            "learning_rate": 0.004,
            "per_device_train_batch_size": 4,  # Smaller batch size
        },
    }

    print(f"🚀 Starting {preset} training demo...")
    print(f"📁 Model: {model_path}")
    print(f"🎯 GPU: {gpu_devices}")
    print(f"📊 Training examples: {len(presets[preset])}")
    print(f"💾 Output: {config['output_dir']}")
    print("-" * 50)

    try:
        response = requests.post(f"{BASE_URL}/api/train", json=config, timeout=30)

        if response.status_code == 200:
            result = response.json()
            print("✅ Training started successfully!")
            print(f"📝 Message: {result.get('message', 'Training initiated')}")

            return monitor_training()

        else:
            error = response.json()
            print(f"❌ Failed to start training: {error.get('error', 'Unknown error')}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Please make sure the EasySteer server is running.")
        print("💡 Start the server with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def monitor_training():
    """Monitor training progress"""
    print("\n📈 Monitoring training progress...")
    print("(Press Ctrl+C to stop monitoring)\n")

    try:
        last_message = None
        started = False
        while True:
            response = requests.get(f"{BASE_URL}/api/train-status", timeout=30)
            response.raise_for_status()

            if response.status_code == 200:
                status = response.json()

                message = status.get("status_message", "")
                if message and message != last_message:
                    print(message)
                    last_message = message

                if status.get("is_training"):
                    started = True
                elif (
                    started
                    or status.get("error_message")
                    or message.startswith("Training complete!")
                ):
                    error = status.get("error_message")
                    if error:
                        print(f"\n❌ Training failed: {error}")
                        return False
                    else:
                        print("\n🎉 Training completed successfully!")
                    return True

            time.sleep(2)  # Check every 2 seconds

    except KeyboardInterrupt:
        print("\n\n⏸️ Monitoring stopped by user.")
    except Exception as e:
        print(f"\n\n❌ Monitoring error: {str(e)}")
    return False


def test_inference(model_name, steer_vector_path, test_inputs, api_url, api_key):
    """Send a trained LoReFT payload to a running vllm-steer server."""
    from easysteer.vectors import from_pyreft, to_json_payload

    payload = from_pyreft(str(steer_vector_path))
    wire = to_json_payload(payload)
    # The adapter also accepts BiasIntervention checkpoints.
    algorithm = {"reft": "loreft", "direction": "direct"}[wire["kind"]]
    spec = {
        "vectors": [
            {
                "data": wire,
                "algorithm": algorithm,
                "scale": 1.0,
                "apply": {"prompt_positions": [-1]},
            }
        ]
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for instruction in test_inputs:
        response = requests.post(
            f"{api_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": instruction}],
                "temperature": 0.0,
                "max_tokens": 128,
                "repetition_penalty": 1.1,
                "steering": spec,
            },
            timeout=180,
        )
        response.raise_for_status()
        generated = response.json()["choices"][0]["message"]["content"]
        print(f"Input: {instruction}\nOutput: {generated}\n")


def main():
    parser = argparse.ArgumentParser(description="EasySteer Training Demo")
    parser.add_argument("--model", required=True, help="Path to the model to train")
    parser.add_argument("--gpu", default="0", help="GPU device IDs (default: 0)")
    parser.add_argument(
        "--preset",
        choices=["emoji", "emotion"],
        default="emoji",
        help="Training preset to use (default: emoji)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run inference against a separately started vllm-steer server after training",
    )
    parser.add_argument(
        "--api-url", default=os.getenv("VLLM_API_URL", "http://localhost:8000/v1")
    )
    parser.add_argument("--api-model", help="Served model name (defaults to --model)")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))

    args = parser.parse_args()

    print("🎯 EasySteer Training Demo")
    print("=" * 50)

    success = start_training_demo(args.model, args.gpu, args.preset)

    if success and args.test:
        steer_vector_path = FRONTEND_DIR / "results" / f"demo_{args.preset}_training"
        test_inputs = [
            "Hello, how are you?",
            "What's the capital of France?",
            "Tell me something interesting",
        ]

        print("\n" + "=" * 50)
        test_inference(
            args.api_model or args.model,
            steer_vector_path,
            test_inputs,
            args.api_url,
            args.api_key,
        )

    if not success:
        raise SystemExit(1)
    print("\n🏁 Demo completed!")


if __name__ == "__main__":
    main()
