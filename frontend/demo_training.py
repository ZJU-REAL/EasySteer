#!/usr/bin/env python3
"""
EasySteer Training Functionality Demo Script

This script demonstrates how to use the training functionality of EasySteer via its API.
"""

import requests
import json
import time
import argparse

# Base URL for the API
BASE_URL = "http://localhost:5000"

def start_training_demo(model_path, gpu_devices="0", preset="emoji"):
    """Start the training demo"""
    
    # Preset training data
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
            ["Tell me about a happy day", "What a joyful and wonderful experience that must have been! 😊"],
            ["I'm feeling sad today", "I understand that you're going through a difficult time. 😔"],
            ["This is so exciting!", "That sounds absolutely thrilling and amazing! 🎉"],
            ["I'm really angry about this", "I can sense your frustration and anger about this situation. 😠"],
            ["I'm worried about tomorrow", "It's completely natural to feel anxious about upcoming events. 😰"],
            ["I love spending time with friends", "Friendship and connection bring such warmth to life! ❤️"],
            ["This is really stressful", "Stress can be overwhelming, and your feelings are valid. 😓"],
            ["I'm proud of my achievement", "You should feel incredibly proud of what you've achieved! 🌟"],
        ]
    }
    
    if preset not in presets:
        print(f"Error: Unknown preset '{preset}'. Available: {list(presets.keys())}")
        return False
    
    # Build the training configuration
    config = {
        "model_path": model_path,
        "gpu_devices": gpu_devices,
        "reft_config": {
            "layer": 8,
            "component": "block_output",
            "low_rank_dimension": 4
        },
        "training_examples": json.dumps(presets[preset]),
        "training_args": {
            "num_train_epochs": 50,  # Fewer epochs for demonstration
            "learning_rate": 0.004,
            "per_device_train_batch_size": 4,  # Smaller batch size
            "output_dir": f"./results/demo_{preset}_training"
        }
    }
    
    print(f"🚀 Starting {preset} training demo...")
    print(f"📁 Model: {model_path}")
    print(f"🎯 GPU: {gpu_devices}")
    print(f"📊 Training examples: {len(presets[preset])}")
    print(f"💾 Output: {config['training_args']['output_dir']}")
    print("-" * 50)
    
    try:
        # Send the training request
        response = requests.post(f"{BASE_URL}/api/train", json=config)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Training started successfully!")
            print(f"📝 Message: {result.get('message', 'Training initiated')}")
            
            # Monitor training progress
            monitor_training()
            
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
    
    return True

def monitor_training():
    """Monitor training progress"""
    print("\n📈 Monitoring training progress...")
    print("(Press Ctrl+C to stop monitoring)\n")
    
    try:
        last_progress = -1
        while True:
            response = requests.get(f"{BASE_URL}/api/train/status")
            
            if response.status_code == 200:
                status = response.json()
                
                progress = status.get('progress', 0)
                is_training = status.get('is_training', False)
                completed = status.get('completed', False)
                error = status.get('error')
                current_loss = status.get('current_loss')
                
                # Only display when progress updates
                if progress != last_progress:
                    progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
                    print(f"\r[{progress_bar}] {progress:.1f}%", end="")
                    
                    if current_loss:
                        print(f" | Loss: {current_loss:.4f}", end="")
                    
                    last_progress = progress
                
                if completed:
                    print("\n\n🎉 Training completed successfully!")
                    output_path = status.get('output_path')
                    if output_path:
                        print(f"💾 Model saved to: {output_path}")
                    break
                elif error:
                    print(f"\n\n❌ Training failed: {error}")
                    break
                elif not is_training:
                    print("\n\n⏹️ Training stopped.")
                    break
                    
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\n⏸️ Monitoring stopped by user.")
    except Exception as e:
        print(f"\n\n❌ Monitoring error: {str(e)}")

def test_inference(model_path, steer_vector_path, test_inputs):
    """Test the trained model"""
    print(f"\n🧪 Testing trained model...")
    print(f"📁 Steer vector: {steer_vector_path}")
    print("-" * 50)
    
    for i, instruction in enumerate(test_inputs, 1):
        config = {
            "model_path": model_path,
            "gpu_devices": "0",
            "normalize_steer_vector": False,
            "instruction": instruction,
            "sampling_params": {
                "temperature": 0.0,
                "max_tokens": 128,
                "repetition_penalty": 1.1
            },
            "steer_vector_name": "demo_trained_model",
            "steer_vector_int_id": 1,
            "steer_vector_local_path": steer_vector_path,
            "scale": 1.0,
            "algorithm": "loreft",
            "target_layers": [8],
            "prefill_trigger_positions": [-1],
            "debug": False
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/generate", json=config)
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get('generated_text', 'No response')
                print(f"🤔 Input {i}: {instruction}")
                print(f"🤖 Output: {generated}")
                print()
            else:
                error = response.json()
                print(f"❌ Test {i} failed: {error.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Test {i} error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="EasySteer Training Demo")
    parser.add_argument("--model", required=True, help="Path to the model to train")
    parser.add_argument("--gpu", default="0", help="GPU device IDs (default: 0)")
    parser.add_argument("--preset", choices=["emoji", "emotion"], default="emoji", 
                       help="Training preset to use (default: emoji)")
    parser.add_argument("--test", action="store_true", 
                       help="Run inference test after training")
    
    args = parser.parse_args()
    
    print("🎯 EasySteer Training Demo")
    print("=" * 50)
    
    # Start training
    success = start_training_demo(args.model, args.gpu, args.preset)
    
    if success and args.test:
        # Test inference
        steer_vector_path = f"./results/demo_{args.preset}_training"
        test_inputs = [
            "Hello, how are you?",
            "What's the capital of France?",
            "Tell me something interesting",
        ]
        
        print("\n" + "="*50)
        test_inference(args.model, steer_vector_path, test_inputs)
    
    print("\n🏁 Demo completed!")

if __name__ == "__main__":
    main() 