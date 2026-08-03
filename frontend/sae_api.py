from flask import Blueprint, request, jsonify
import os
import logging

from config import SAE_PARAMS_PATH
from core import PROJECT_ROOT, lang, project_root_on_path, require_fields

# Import repo-local SAE modules without permanently mutating sys.path
with project_root_on_path():
    from easysteer.steer.sae import (
        search_sae_features,
        get_sae_feature_explanation,
        extract_sae_decoder_vector
    )

# Create blueprint for SAE-related endpoints
sae_bp = Blueprint('sae', __name__)

# Configure logging
logger = logging.getLogger(__name__)

@sae_bp.route('/api/sae/search', methods=['POST'])
def search_features():
    """API endpoint to search SAE features"""
    try:
        data = request.json

        # Validate required fields
        error = require_fields(data, ['model_id', 'sae_id', 'query'], lang(request))
        if error:
            return jsonify({'error': error}), 400

        # Search for SAE features
        results = search_sae_features(
            model_id=data['model_id'],
            sae_id=data['sae_id'],
            query=data['query'],
            api_key=data.get('api_key')
        )

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        logger.error(f"Error in search_features endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@sae_bp.route('/api/sae/feature/<model_id>/<sae_id>/<int:feature_index>', methods=['GET'])
def get_feature_details(model_id, sae_id, feature_index):
    """API endpoint to get details for a specific SAE feature"""
    try:
        # Get the API key from query parameter or default to None
        api_key = request.args.get('api_key')

        # Get feature explanation
        result = get_sae_feature_explanation(
            model_id=model_id,
            sae_id=sae_id,
            feature_index=feature_index,
            api_key=api_key
        )

        return jsonify({
            'success': True,
            'feature': result
        })

    except Exception as e:
        logger.error(f"Error in get_feature_details endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@sae_bp.route('/api/sae/extract-vector', methods=['POST'])
def extract_sae_vector():
    """API endpoint to extract SAE feature vector and save as a steering vector"""
    try:
        data = request.json

        # Validate required fields
        error = require_fields(data, ['feature_index', 'vector_name'], lang(request))
        if error:
            return jsonify({'error': error}), 400

        feature_index = data['feature_index']
        vector_name = data['vector_name']
        scale = data.get('scale', 1.0)

        # The SAE decoder weights file is configured via the environment
        if not SAE_PARAMS_PATH:
            return jsonify({
                'success': False,
                'error': (
                    'SAE_PARAMS_PATH is not set. Export SAE_PARAMS_PATH='
                    '/path/to/params.npz (the SAE decoder weights, e.g. a '
                    'gemma-scope params.npz) before starting the server.'
                )
            }), 500

        # Create vectors directory if it doesn't exist
        vectors_dir = os.path.join(PROJECT_ROOT, 'vectors')
        if not os.path.exists(vectors_dir):
            os.makedirs(vectors_dir)

        # Create output filename using feature ID
        vector_filename = f"{feature_index}.pt"
        output_path = os.path.join(vectors_dir, vector_filename)

        # Try to extract the decoder vector
        logger.info(f"Extracting SAE vector for feature {feature_index} from {SAE_PARAMS_PATH}")
        vector = extract_sae_decoder_vector(SAE_PARAMS_PATH, feature_index, output_path)

        if vector is None:
            return jsonify({
                'success': False,
                'error': 'Failed to extract vector, check server logs for details'
            }), 500

        # Return success response with vector info
        return jsonify({
            'success': True,
            'vector': {
                'name': vector_name,
                'feature_index': feature_index,
                'file_path': output_path,
                'scale': scale
            }
        })

    except Exception as e:
        logger.error(f"Error in extract_sae_vector endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
