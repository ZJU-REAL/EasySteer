// 推理功能模块

import { parseListInput, escapeHtml } from './utils.js';

// 显示响应结果
export function showResponse(data) {
    const responseDiv = document.getElementById('response');
    const responseContent = document.getElementById('responseContent');
    const errorDiv = document.getElementById('error');
    
    // Format and display the generation result
    if (data.generated_text) {
        // Check if we have both baseline and steered text for comparison
        if (data.baseline_text) {
            // Create a vertical comparison view with responses stacked (baseline first)
            const comparisonHTML = `
                <div class="response-comparison-vertical">
                    <div class="response-section baseline-section">
                        <h4><i class="fas fa-robot"></i> ${window.t('baseline_title')}</h4>
                        <pre class="baseline-text">${escapeHtml(data.baseline_text)}</pre>
                    </div>
                    <div class="response-section steered-section">
                        <h4><i class="fas fa-magic"></i> ${window.t('steered_title')}</h4>
                        <pre class="steered-text">${escapeHtml(data.generated_text)}</pre>
                    </div>
                </div>
            `;
            responseContent.innerHTML = comparisonHTML;
        } else {
            // Regular single response
            responseContent.innerHTML = `<pre>${escapeHtml(data.generated_text)}</pre>`;
        }
    } else {
        // For non-generation responses (messages, etc.)
        responseContent.innerHTML = `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
    }
    
    responseDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    
    // Scroll to the result
    responseDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display error message
export function showError(message) {
    const errorDiv = document.getElementById('error');
    const errorContent = document.getElementById('errorContent');
    const responseDiv = document.getElementById('response');
    
    errorContent.textContent = message;
    errorDiv.style.display = 'block';
    responseDiv.style.display = 'none';
    
    // Scroll to the error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Submit configuration
export async function submitConfiguration() {
    const config = {
        // Model configuration
        model_path: document.getElementById('modelPath').value,
        gpu_devices: document.getElementById('gpuDevices').value,
        instruction: document.getElementById('instruction').value,
        
        // Sampling parameters
        sampling_params: {
            temperature: parseFloat(document.getElementById('temperature').value),
            max_tokens: parseInt(document.getElementById('maxTokens').value),
            repetition_penalty: parseFloat(document.getElementById('repetitionPenalty').value)
        },
        
        // Steer Vector configuration
        steer_vector_name: document.getElementById('steerVectorName').value,
        steer_vector_int_id: parseInt(document.getElementById('steerVectorId').value),
        steer_vector_local_path: document.getElementById('localPath').value,
        scale: parseFloat(document.getElementById('scale').value),
        algorithm: document.getElementById('algorithm').value,
        target_layers: parseListInput(document.getElementById('targetLayers').value),
        prefill_trigger_tokens: parseListInput(document.getElementById('prefillTriggerTokens').value),
        prefill_trigger_positions: parseListInput(document.getElementById('prefillTriggerPositions').value),
        generate_trigger_tokens: parseListInput(document.getElementById('generateTriggerTokens').value)
    };

    // Validate required fields
    if (!config.steer_vector_name || !config.steer_vector_int_id || !config.steer_vector_local_path || !config.model_path || !config.instruction) {
        showError(window.t('required_fields_error'));
        return;
    }

    // Show loading state
    const submitButton = document.querySelector('.btn-primary');
    const originalHTML = submitButton.innerHTML;
    submitButton.innerHTML = '<span class="loading"></span> ' + window.t('generating');
    submitButton.disabled = true;

    try {
        // Send request to the backend - auto-detect current host
        const apiUrl = `${window.location.protocol}//${window.location.hostname}:5000/api/generate`;
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept-Language': window.currentLanguage
            },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (response.ok) {
            showResponse(data);
        } else {
            showError(data.error || window.t('submit_failed'));
        }
    } catch (error) {
        console.error('Error:', error);
        showError(window.t('network_error'));
    } finally {
        // Restore button state
        submitButton.innerHTML = originalHTML;
        submitButton.disabled = false;
    }
}

// Reset form
export function resetForm() {
    // Model configuration
    document.getElementById('modelPath').value = '';
    document.getElementById('gpuDevices').value = '0';
    document.getElementById('instruction').value = '';
    
    // Sampling parameters
    document.getElementById('temperature').value = '0.0';
    document.getElementById('temperatureSlider').value = '0.0';
    document.getElementById('maxTokens').value = '128';
    document.getElementById('repetitionPenalty').value = '1.1';
    document.getElementById('repetitionPenaltySlider').value = '1.1';
    
    // Steer Vector configuration
    document.getElementById('steerVectorName').value = '';
    document.getElementById('steerVectorId').value = '';
    document.getElementById('localPath').value = '';
    document.getElementById('scale').value = '1.0';
    document.getElementById('scaleSlider').value = '1.0';
    document.getElementById('algorithm').value = 'direct';
    document.getElementById('targetLayers').value = '';
    document.getElementById('prefillTriggerTokens').value = '';
    document.getElementById('prefillTriggerPositions').value = '';
    document.getElementById('generateTriggerTokens').value = '';
    document.getElementById('fileInput').value = '';
    
    // Hide results and errors
    document.getElementById('response').style.display = 'none';
    document.getElementById('error').style.display = 'none';
}

// Dynamically load configuration options
export async function loadConfigOptions() {
    try {
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/configs`);
        if (response.ok) {
            const data = await response.json();
            const configSelect = document.getElementById('configSelect');
            
            // Clear existing options (except the default one)
            while (configSelect.children.length > 1) {
                configSelect.removeChild(configSelect.lastChild);
            }
            
            // Add dynamically loaded configuration options (single vector configs only)
            data.configs
                .filter(config => config.type === 'single_vector')
                .forEach(config => {
                    const option = document.createElement('option');
                    option.value = config.name;
                    option.textContent = config.display_name;
                    configSelect.appendChild(option);
                });
        }
    } catch (error) {
        console.log('Could not load configuration options, using default options');
    }
}

// Import selected configuration
export async function importSelectedConfig() {
    const configSelect = document.getElementById('configSelect');
    const selectedConfig = configSelect.value;
    
    if (!selectedConfig) {
        showError(window.t('error_select_config'));
        return;
    }
    
    try {
        window.showStatus(window.t('importing_config', { configName: configSelect.options[configSelect.selectedIndex].text }), 'info');
        
        // Get config file from the backend
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/config/${selectedConfig}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const config = await response.json();
        
        // Set model configuration
        document.getElementById('modelPath').value = config.model.path || '';
        document.getElementById('gpuDevices').value = config.model.gpu_devices || '';
        document.getElementById('instruction').value = config.model.instruction || '';
        
        // Set sampling parameters
        document.getElementById('temperature').value = config.sampling.temperature || '0.0';
        document.getElementById('temperatureSlider').value = config.sampling.temperature || '0.0';
        document.getElementById('maxTokens').value = config.sampling.max_tokens || '128';
        document.getElementById('repetitionPenalty').value = config.sampling.repetition_penalty || '1.1';
        document.getElementById('repetitionPenaltySlider').value = config.sampling.repetition_penalty || '1.1';
        
        // Set Steer Vector configuration
        document.getElementById('steerVectorName').value = config.steer_vector.name || '';
        document.getElementById('steerVectorId').value = config.steer_vector.id || '';
        document.getElementById('localPath').value = config.steer_vector.path || '';
        document.getElementById('scale').value = config.steer_vector.scale || '1.0';
        document.getElementById('scaleSlider').value = config.steer_vector.scale || '1.0';
        document.getElementById('algorithm').value = config.steer_vector.algorithm || 'direct';
        document.getElementById('targetLayers').value = config.steer_vector.target_layers || '';
        
        // Set trigger configuration - based on algorithm type
        if (config.steer_vector.algorithm === 'loreft') {
            // LoReft algorithm uses position triggers
            document.getElementById('prefillTriggerPositions').value = config.steer_vector.prefill_positions || '';
            document.getElementById('prefillTriggerTokens').value = '';
            document.getElementById('generateTriggerTokens').value = '';
        } else {
            // Direct algorithm uses token triggers
            document.getElementById('prefillTriggerTokens').value = config.steer_vector.prefill_trigger_tokens || '';
            document.getElementById('generateTriggerTokens').value = config.steer_vector.generate_trigger_tokens || '';
            document.getElementById('prefillTriggerPositions').value = '';
        }
        
        showResponse({
            message: window.t('import_success_message'),
            description: window.t('import_success_description', { configName: configSelect.options[configSelect.selectedIndex].text })
        });
        
        // Clear selection box
        configSelect.value = '';
        
    } catch (error) {
        showError(window.t('import_fail_error') + ': ' + error.message);
    }
} 