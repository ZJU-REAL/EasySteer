"""Search SAE features and extract their decoder vectors."""

import logging
import os
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)


class SAEFeatureExplorer:
    """Explore Neuronpedia features and local SAE decoder weights."""

    def __init__(self, api_key: str | None = None):
        """Configure access to Neuronpedia.

        Args:
            api_key: Neuronpedia API key; defaults to NP_API_KEY.
        """
        self.api_key = api_key or os.environ.get("NP_API_KEY")

        if not self.api_key:
            logger.warning(
                "No Neuronpedia API key provided. Set NP_API_KEY environment variable or pass to constructor."
            )

    def search_features(
        self, model_id: str, sae_id: str, query: str
    ) -> list[dict[str, Any]]:
        """Search SAE features by semantic query.

        Args:
            model_id: Model identifier (e.g., 'gemma-2-9b')
            sae_id: SAE identifier (e.g., '24-gemmascope-res-16k')
            query: Search query

        Returns:
            List of matching features sorted by relevance
        """
        try:
            url = "https://www.neuronpedia.org/api/explanation/search"
            payload = {"modelId": model_id, "layers": [sae_id], "query": query}

            headers = {"Content-Type": "application/json", "X-Api-Key": self.api_key}

            logger.info(f"Searching for features related to '{query}'...")
            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                results = response.json()

                filtered_results = []
                for result in results.get("results", []):
                    filtered_result = {
                        "modelId": result.get("modelId"),
                        "layer": result.get("layer"),
                        "index": result.get("index"),
                        "description": result.get("description"),
                        "explanationModelName": result.get("explanationModelName"),
                        "typeName": result.get("typeName"),
                        "cosine_similarity": result.get("cosine_similarity"),
                    }
                    filtered_results.append(filtered_result)

                filtered_results.sort(
                    key=lambda x: x.get("cosine_similarity", 0), reverse=True
                )

                logger.info(f"Found {len(filtered_results)} related features")
                return filtered_results
            raise RuntimeError(
                f"Neuronpedia search failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        except requests.RequestException as e:
            raise RuntimeError(f"Neuronpedia is unreachable: {e}") from e

    def get_feature_explanation(
        self, model_id: str, sae_id: str, feature_index: int
    ) -> dict[str, Any]:
        """Fetch a feature's explanation, token scores, and activation example.

        Args:
            model_id: Model identifier (e.g., 'gemma-2-9b')
            sae_id: SAE identifier (e.g., '24-gemmascope-res-16k')
            feature_index: Feature index number

        Returns:
            Dictionary containing processed feature explanation details
        """
        try:
            url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_index}"

            headers = {"Content-Type": "application/json", "X-Api-Key": self.api_key}

            logger.info(f"Fetching explanation for feature index {feature_index}...")
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                raw_data = response.json()

                processed_data = {
                    "basic_info": {
                        "modelId": raw_data.get("modelId"),
                        "layer": raw_data.get("layer"),
                        "index": raw_data.get("index"),
                    },
                    "explanation": None,
                    "sparsity": raw_data.get("frac_nonzero"),
                    "top_activating_tokens": [],
                    "top_inhibiting_tokens": [],
                    "activation_example": None,
                }

                explanations = raw_data.get("explanations", [])
                if explanations:
                    processed_data["explanation"] = explanations[0].get("description")

                pos_str = raw_data.get("pos_str", [])
                pos_values = raw_data.get("pos_values", [])
                for token, value in zip(pos_str[:5], pos_values[:5]):
                    processed_data["top_activating_tokens"].append(
                        {"token": token, "activation_value": float(value)}
                    )

                neg_str = raw_data.get("neg_str", [])
                neg_values = raw_data.get("neg_values", [])
                for token, value in zip(neg_str[:5], neg_values[:5]):
                    processed_data["top_inhibiting_tokens"].append(
                        {"token": token, "activation_value": float(value)}
                    )

                activations = raw_data.get("activations", [])
                if activations:
                    first_activation = activations[0]
                    max_value = first_activation.get("maxValue", 0)
                    max_value_token_index = first_activation.get("maxValueTokenIndex")
                    all_tokens = first_activation.get("tokens", [])

                    if max_value_token_index is not None and all_tokens:
                        trigger_token = all_tokens[max_value_token_index]

                        context_window = 7
                        start_index = max(0, max_value_token_index - context_window)
                        end_index = min(
                            len(all_tokens), max_value_token_index + context_window + 1
                        )

                        context_text = "".join(
                            all_tokens[start_index:end_index]
                        ).replace("\u2581", " ")

                        processed_data["activation_example"] = {
                            "max_value": float(max_value),
                            "trigger_token": trigger_token,
                            "context": context_text,
                        }

                return processed_data
            raise RuntimeError(
                f"Neuronpedia feature lookup failed with HTTP "
                f"{response.status_code}: {response.text[:200]}"
            )

        except requests.RequestException as e:
            raise RuntimeError(f"Neuronpedia is unreachable: {e}") from e

    def extract_decoder_vector(
        self,
        model_file: str,
        feature_index: int,
        save_path: str | None = None,
        return_vector: bool = True,
    ) -> np.ndarray | None:
        """Extract one feature's decoder vector from an SAE checkpoint.

        Args:
            model_file: Path to the SAE model file (npz format)
            feature_index: Feature index to extract
            save_path: Optional path to save the vector as PyTorch file (.pt)
            return_vector: Whether to return the vector as numpy array

        Returns:
            Decoder vector if return_vector is True and extraction succeeds;
            otherwise None. Extraction failures are logged.
        """
        if save_path:
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "Saving an SAE decoder vector as a PyTorch file requires torch; "
                    "install PyTorch or omit save_path to receive a NumPy array"
                ) from exc

        try:
            if not os.path.exists(model_file):
                logger.error(f"Model file not found: {model_file}")
                return None

            logger.info(f"Loading SAE model from: {model_file}")
            data = np.load(model_file)

            if "W_dec" not in data:
                logger.error("Decoder weights ('W_dec') not found in the model file")
                return None

            W_dec = data["W_dec"]

            if feature_index < 0 or feature_index >= W_dec.shape[0]:
                logger.error(
                    f"Invalid feature index: {feature_index}. Valid range: 0 to {W_dec.shape[0] - 1}"
                )
                return None

            feature_vector = W_dec[feature_index, :]

            logger.info(
                f"Extracted decoder vector for feature {feature_index}, shape: {feature_vector.shape}"
            )

            if save_path:
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch_vector = torch.tensor(feature_vector)
                torch.save(torch_vector, save_path)
                logger.info(f"Saved decoder vector to: {save_path}")

            return feature_vector if return_vector else None

        except Exception as e:
            logger.error(f"Error extracting decoder vector: {e}")
            return None


def search_sae_features(
    model_id: str, sae_id: str, query: str, api_key: str | None = None
) -> list[dict[str, Any]]:
    """Call SAEFeatureExplorer.search_features without retaining a client.

    api_key defaults to the NP_API_KEY environment variable.
    """
    explorer = SAEFeatureExplorer(api_key=api_key)
    return explorer.search_features(model_id, sae_id, query)


def get_sae_feature_explanation(
    model_id: str, sae_id: str, feature_index: int, api_key: str | None = None
) -> dict[str, Any]:
    """Call SAEFeatureExplorer.get_feature_explanation with a temporary client.

    api_key defaults to the NP_API_KEY environment variable.
    """
    explorer = SAEFeatureExplorer(api_key=api_key)
    return explorer.get_feature_explanation(model_id, sae_id, feature_index)


def extract_sae_decoder_vector(
    model_file: str, feature_index: int, save_path: str | None = None
) -> np.ndarray | None:
    """Call SAEFeatureExplorer.extract_decoder_vector and return its array."""
    explorer = SAEFeatureExplorer()
    return explorer.extract_decoder_vector(model_file, feature_index, save_path)
