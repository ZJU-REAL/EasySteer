from flask import Blueprint, request, jsonify
import os
import threading
import logging
import json
import time

from transformers.trainer_callback import TrainerCallback

from core import (
    ConfigStore,
    get_message,
    lang,
    project_root_on_path,
    require_fields,
    resource_manager,
)

# Create a blueprint for training-related endpoints
training_bp = Blueprint('training', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Global training status tracking
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'current_step': 0,
    'status_message': '',
    'error_message': '',
    'logs': []
}

# Config presets served by /api/train-configs and /api/train-config/<name>
config_store = ConfigStore(
    'training',
    display_names={
        'emoji_loreft': 'Emoji LoReft Training Configuration',
        'emoji_bias': 'Emoji Bias Training Configuration',
    },
)


class TrainingProgressCallback(TrainerCallback):
    """Custom training callback to track progress"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when a log is created during training"""
        global training_status

        if logs:
            # Update training status
            training_status['current_step'] = state.global_step
            training_status['current_epoch'] = state.epoch

            # Format the log message
            log_message = f"[{time.strftime('%H:%M:%S')}] "

            # Process all possible log fields
            log_parts = []

            if 'epoch' in logs:
                log_parts.append(f"Epoch: {logs['epoch']:.2f}")

            # Prioritize loss-related information
            if 'loss' in logs:
                log_parts.append(f"Loss: {logs['loss']:.4f}")
            elif 'train_loss' in logs:
                log_parts.append(f"Loss: {logs['train_loss']:.4f}")

            if 'grad_norm' in logs:
                log_parts.append(f"Grad: {logs['grad_norm']:.4f}")

            if 'learning_rate' in logs:
                log_parts.append(f"LR: {logs['learning_rate']:.2e}")

            if 'train_runtime' in logs:
                log_parts.append(f"Runtime: {logs['train_runtime']:.2f}s")

            if 'train_samples_per_second' in logs:
                log_parts.append(f"Speed: {logs['train_samples_per_second']:.2f} samples/s")

            if 'eval_loss' in logs:
                log_parts.append(f"Eval Loss: {logs['eval_loss']:.4f}")

            # Assemble the complete log message
            if log_parts:
                log_message += " | ".join(log_parts)
            else:
                # If no recognized fields, display the raw log
                log_message += str(logs)

            # Add to the log list
            if len(training_status['logs']) > 100:  # Keep only the last 100 logs
                training_status['logs'] = training_status['logs'][-50:]

            training_status['logs'].append(log_message)

            # Update status message
            if 'loss' in logs:
                training_status['status_message'] = f"Training - Step {state.global_step}, Loss: {logs['loss']:.4f}"
            elif 'train_loss' in logs:
                training_status['status_message'] = f"Training - Step {state.global_step}, Loss: {logs['train_loss']:.4f}"


@training_bp.route('/api/train', methods=['POST'])
def train():
    """Start training"""
    try:
        data = request.json
        request_lang = lang(request)

        # Validate required fields
        error = require_fields(data, ['model_path', 'training_examples'], request_lang)
        if error:
            return jsonify({'error': error}), 400

        # output_dir may arrive top-level (web UI) or inside training_args (demo script)
        output_dir = data.get('output_dir') or data.get('training_args', {}).get('output_dir')
        if not output_dir:
            return jsonify({'error': get_message('missing_field', request_lang, field='output_dir')}), 400

        # Validate training data format (the web UI sends it as a JSON string)
        try:
            training_examples = data['training_examples']
            if isinstance(training_examples, str):
                training_examples = json.loads(training_examples)
            if not isinstance(training_examples, list) or len(training_examples) == 0:
                return jsonify({'error': 'Training data must be a non-empty array'}), 400

            for i, example in enumerate(training_examples):
                if not isinstance(example, list) or len(example) != 2:
                    return jsonify({'error': f'Training example {i} has incorrect format. Must be an array of two elements [input, output]'}), 400
        except Exception as e:
            return jsonify({'error': f'Incorrect training data format: {str(e)}'}), 400

        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = data.get('gpu_devices', '0')

        # Start training (using asynchronous method)
        def train_model():
            global training_status
            try:
                # Initialize training status
                training_status.update({
                    'is_training': True,
                    'current_epoch': 0,
                    'current_step': 0,
                    'status_message': 'Initializing training...',
                    'error_message': '',
                    'logs': []
                })

                # Import the shared training pipeline from the local easysteer package
                with project_root_on_path():
                    from easysteer.reft.train import train_reft

                logger.info(f"Starting to load model: {data['model_path']}")
                training_status['status_message'] = f"Loading model: {data['model_path']}"

                reft_config = data.get('reft_config', {})
                training_args = data.get('training_args', {})

                train_reft(
                    model_path=data['model_path'],
                    examples=training_examples,
                    intervention=data.get('intervention', 'loreft'),
                    layer=reft_config.get('layer', 8),
                    component=reft_config.get('component', 'block_output'),
                    low_rank_dimension=reft_config.get('low_rank_dimension', 4),
                    callbacks=[TrainingProgressCallback()],
                    save_dir=output_dir,
                    output_dir=output_dir,
                    num_train_epochs=training_args.get('num_train_epochs', 100.0),
                    per_device_train_batch_size=training_args.get('per_device_train_batch_size', 10),
                    learning_rate=training_args.get('learning_rate', 4e-3),
                    logging_steps=training_args.get('logging_steps', 40),
                )

                # Training finished
                training_status.update({
                    'is_training': False,
                    'status_message': f"Training complete! Model saved to: {output_dir}"
                })

                logger.info(f"Training complete, model saved to: {output_dir}")

            except Exception as e:
                # Training failed
                training_status.update({
                    'is_training': False,
                    'error_message': str(e),
                    'status_message': f"Training failed: {str(e)}"
                })
                logger.exception(f"Training failed: {str(e)}")

        # Start training in a background thread
        train_thread = threading.Thread(target=train_model)
        train_thread.daemon = True
        train_thread.start()

        return jsonify({
            'success': True,
            'message': 'Training has started',
            'output_dir': output_dir,
            'training_examples_count': len(training_examples),
            'reft_config': data.get('reft_config', {}),
            'training_args': data.get('training_args', {}),
            'note': 'Training is running in the background. Check server logs for progress.'
        }), 200

    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        return jsonify({'error': get_message('server_error', lang(request), error=str(e))}), 500


@training_bp.route('/api/train-configs', methods=['GET'])
def list_train_configs():
    """List all available training configuration files"""
    try:
        return jsonify({"configs": config_store.list()})

    except Exception as e:
        logger.error(f"Failed to list training configs: {e}")
        return jsonify({"error": f"Failed to list training configs: {str(e)}"}), 500


@training_bp.route('/api/train-config/<config_name>', methods=['GET'])
def get_train_config(config_name):
    """Get a training configuration file"""
    try:
        config = config_store.get(config_name)
        if config is None:
            return jsonify({"error": f"Training config {config_name} not found"}), 404

        return jsonify(config)

    except Exception as e:
        logger.error(f"Failed to get training config: {e}")
        return jsonify({"error": f"Failed to get training config: {str(e)}"}), 500


@training_bp.route('/api/train-status', methods=['GET'])
def get_train_status():
    """Get training status"""
    global training_status
    return jsonify(training_status), 200


@training_bp.route('/api/train-restart', methods=['POST'])
def restart_training_backend():
    """Fully restart the training backend process with proper GPU memory cleanup."""
    try:
        global training_status
        training_status.update({
            'is_training': False,
            'status_message': 'Preparing to fully restart the backend process...',
            'logs': []
        })

        result = resource_manager.restart_backend(delay=1.0)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Failed to restart backend: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Failed to restart backend: {str(e)}"
        }), 500
