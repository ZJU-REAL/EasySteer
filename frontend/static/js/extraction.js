// 提取功能模块

// Extraction status polling interval ID
let extractionStatusInterval = null;
// Keep track of sample pairs count
let samplePairsCount = 0;

// Add a new sample pair
export function addSamplePair(positiveText = '', negativeText = '') {
    // Get the template
    const template = document.getElementById('samplePairTemplate');
    const samplePairsList = document.getElementById('samplePairsList');
    
    // Remove empty message if it exists
    const emptyMessage = samplePairsList.querySelector('.empty-samples-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    // Clone the template content
    const clone = template.content.cloneNode(true);
    
    // Set unique index and update content
    const samplePairItem = clone.querySelector('.sample-pair-item');
    const index = samplePairsCount++;
    samplePairItem.dataset.index = index;
    
    // Set the sample number
    clone.querySelector('.sample-number').textContent = `#${index + 1}`;
    
    // Set remove button action
    const removeBtn = clone.querySelector('.remove-sample-btn');
    removeBtn.onclick = function() {
        removeSamplePair(index);
    };
    
    // Set input values if provided
    if (positiveText) {
        clone.querySelector('.sample-positive-text').value = positiveText;
    }
    if (negativeText) {
        clone.querySelector('.sample-negative-text').value = negativeText;
    }
    
    // Add change event listeners
    const positiveTextarea = clone.querySelector('.sample-positive-text');
    const negativeTextarea = clone.querySelector('.sample-negative-text');
    positiveTextarea.onchange = updateSamplePairs;
    negativeTextarea.onchange = updateSamplePairs;
    
    // Append to the list
    samplePairsList.appendChild(clone);
    
    // Update the hidden fields
    updateSamplePairs();
    
    // Return the index of the added pair
    return index;
}

// Remove a sample pair
export function removeSamplePair(index) {
    const samplePairsList = document.getElementById('samplePairsList');
    const samplePair = samplePairsList.querySelector(`.sample-pair-item[data-index="${index}"]`);
    
    if (samplePair) {
        samplePair.remove();
        
        // Re-number remaining sample pairs
        const samplePairs = samplePairsList.querySelectorAll('.sample-pair-item');
        samplePairs.forEach((pair, i) => {
            // Update the display number
            pair.querySelector('.sample-number').textContent = `#${i + 1}`;
            // Update the data-index attribute
            pair.dataset.index = i;
            // Update the remove button onclick handler
            const removeBtn = pair.querySelector('.remove-sample-btn');
            removeBtn.onclick = function() {
                removeSamplePair(i);
            };
        });
        
        // Show empty message if no pairs left
        if (samplePairs.length === 0) {
            const emptyMessage = document.createElement('div');
            emptyMessage.className = 'empty-samples-message';
            emptyMessage.setAttribute('data-i18n', 'no_samples_message');
            emptyMessage.textContent = window.t ? window.t('no_samples_message') : 'No sample pairs added yet. Click "Add Sample Pair" to add your first pair.';
            samplePairsList.appendChild(emptyMessage);
        }
        
        // Update the samplePairsCount to match the current number of samples
        samplePairsCount = samplePairs.length;
        
        // Update the hidden fields
        updateSamplePairs();
    }
}

// Update the hidden fields with current samples
export function updateSamplePairs() {
    const samplePairsList = document.getElementById('samplePairsList');
    const samplePairs = samplePairsList.querySelectorAll('.sample-pair-item');
    
    const positiveSamples = [];
    const negativeSamples = [];
    
    samplePairs.forEach(pair => {
        const positiveText = pair.querySelector('.sample-positive-text').value.trim();
        const negativeText = pair.querySelector('.sample-negative-text').value.trim();
        
        if (positiveText) {
            positiveSamples.push(positiveText);
        }
        
        if (negativeText) {
            negativeSamples.push(negativeText);
        }
    });
    
    // Update hidden fields
    document.getElementById('extractPositiveSamples').value = positiveSamples.join('\n');
    document.getElementById('extractNegativeSamples').value = negativeSamples.join('\n');
    
    return { positiveSamples, negativeSamples };
}

// Update extraction method options
export function updateExtractionMethodOptions() {
    const method = document.getElementById('extractMethod').value;
    
    // 只保留DiffMean和PCA方法，不再需要显示SAE选项
    // 保留此函数以防将来需要添加其他方法选项
}

// Display extraction response
export function showExtractResponse(data) {
    const responseDiv = document.getElementById('extractResponse');
    const responseContent = document.getElementById('extractResponseContent');
    const errorDiv = document.getElementById('extractError');
    
    if (data.success) {
        // Create extraction progress HTML
        responseContent.innerHTML = createExtractionProgressHTML();
        startExtractionStatusPolling();
    } else {
        // Display detailed results
        let resultHTML = '<div class="extraction-result">';
        resultHTML += '<div class="result-info">';
        resultHTML += `<p><strong>${window.t('status_label')}:</strong> ${data.message || window.t('extraction_complete')}</p>`;
        if (data.output_path) {
            resultHTML += `<p><strong>${window.t('output_file_label')}:</strong> <code>${data.output_path}</code></p>`;
        }
        if (data.metadata) {
            resultHTML += `<p><strong>${window.t('metadata_label')}:</strong></p>`;
            resultHTML += '<pre>' + JSON.stringify(data.metadata, null, 2) + '</pre>';
        }
        resultHTML += '</div>';
        resultHTML += '</div>';
        
        responseContent.innerHTML = resultHTML;
    }
    
    responseDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    
    // Scroll to the result
    responseDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display extraction error message
export function showExtractError(message) {
    const errorDiv = document.getElementById('extractError');
    const errorContent = document.getElementById('extractErrorContent');
    const responseDiv = document.getElementById('extractResponse');
    
    errorContent.textContent = message;
    errorDiv.style.display = 'block';
    responseDiv.style.display = 'none';
    
    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Start extraction
export async function startExtraction() {
    // First update samples to ensure the latest data
    updateSamplePairs();
    
    // Get form data
    const extractConfig = {
        // Model configuration
        model_path: document.getElementById('extractModelPath').value,
        gpu_devices: document.getElementById('extractGpuDevices').value,
        
        // Extraction method configuration
        method: document.getElementById('extractMethod').value,
        target_layer: document.getElementById('extractTargetLayer').value ? 
            parseInt(document.getElementById('extractTargetLayer').value) : null,
        token_pos: document.getElementById('extractTokenPos').value,
        normalize: document.getElementById('extractNormalize').checked,
        
        // Sample data - get from hidden fields
        positive_samples: document.getElementById('extractPositiveSamples').value.trim().split('\n').filter(s => s.trim()),
        negative_samples: document.getElementById('extractNegativeSamples').value.trim().split('\n').filter(s => s.trim()),
        
        // Output configuration
        output_path: document.getElementById('extractOutputPath').value,
        vector_name: document.getElementById('extractVectorName').value
    };

    // DiffMean和PCA方法不需要额外配置

    // Validate required fields
    if (!extractConfig.model_path || !extractConfig.output_path) {
        showExtractError(window.t('required_fields_error'));
        return;
    }

    // Check if there are any samples
    if (extractConfig.positive_samples.length === 0 || extractConfig.negative_samples.length === 0) {
        showExtractError(window.t ? window.t('no_samples_error') : 'Please add at least one positive and one negative sample.');
        return;
    }

    // Show loading state
    const extractButton = document.querySelector('[onclick="startExtraction()"]');
    const originalHTML = extractButton.innerHTML;
    extractButton.innerHTML = '<span class="loading"></span> ' + window.t('extracting_in_progress');
    extractButton.disabled = true;

    try {
        // Send request to backend
        const apiUrl = window.EasySteerConfig.getApiUrl('/api/extract');
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept-Language': window.currentLanguage
            },
            body: JSON.stringify(extractConfig)
        });

        const data = await response.json();

        if (response.ok) {
            showExtractResponse(data);
        } else {
            showExtractError(data.error || window.t('extraction_failed_error'));
        }
    } catch (error) {
        console.error('Extraction error:', error);
        showExtractError(window.t('network_error'));
    } finally {
        // Restore button state
        extractButton.innerHTML = originalHTML;
        extractButton.disabled = false;
    }
}

// Create extraction progress HTML
function createExtractionProgressHTML() {
    return `
        <div class="logs-container" id="extractionLogs">
            <div class="log-entry status-entry">${window.t('initializing_extraction')}</div>
        </div>
    `;
}

// Start polling for extraction status
function startExtractionStatusPolling() {
    // Clear previous polling
    if (extractionStatusInterval) {
        clearInterval(extractionStatusInterval);
    }
    
    // Get status immediately once
    updateExtractionStatus();
    
    // Poll every 2 seconds
    extractionStatusInterval = setInterval(updateExtractionStatus, 2000);
}

// Stop polling for extraction status
function stopExtractionStatusPolling() {
    if (extractionStatusInterval) {
        clearInterval(extractionStatusInterval);
        extractionStatusInterval = null;
    }
}

// Update extraction status
async function updateExtractionStatus() {
    try {
        const response = await fetch(window.EasySteerConfig.getApiUrl('/api/extract-status'));
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const status = await response.json();
        displayExtractionStatus(status);
        
        // If extraction is finished, stop polling
        if (!status.is_extracting) {
            stopExtractionStatusPolling();
        }
        
    } catch (error) {
        console.error('Failed to get extraction status:', error);
        // Stop polling on network error
        stopExtractionStatusPolling();
    }
}

// Display extraction status
function displayExtractionStatus(status) {
    const logsContainer = document.getElementById('extractionLogs');
    if (!logsContainer) return;
    
    // Create log content array
    let allLogs = [];
    
    // Add status message
    if (status.status_message) {
        let statusClass = 'status-entry';
        
        if (status.error_message) {
            statusClass += ' error-status';
        } else if (!status.is_extracting && status.status_message.includes('完成')) {
            statusClass += ' success-status';
        } else if (status.is_extracting) {
            statusClass += ' training-status';
        }
        
        allLogs.push(`<div class="log-entry ${statusClass}">
            <strong>📊 ${status.status_message}</strong>
        </div>`);
    }
    
    // Add extraction logs
    if (status.logs && status.logs.length > 0) {
        const formattedLogs = status.logs.slice(-20).map(log => {
            return `<div class="log-entry">${log}</div>`;
        });
        
        allLogs = allLogs.concat(formattedLogs);
    }
    
    // Add result info
    if (status.result) {
        allLogs.push(`<div class="log-entry status-entry success-status"><strong>✅ ${window.t('extraction_complete')}</strong></div>`);
        if (status.result.output_path) {
            allLogs.push(`<div class="log-entry">📁 ${window.t('output_file_label')}: <code>${status.result.output_path}</code></div>`);
        }
        if (status.result.layers_extracted) {
            allLogs.push(`<div class="log-entry">📊 ${window.t('layers_extracted_label')}: ${status.result.layers_extracted}</div>`);
        }
    }
    
    // If there are no logs, show waiting message
    if (allLogs.length === 0) {
        allLogs.push(`<div class="log-entry status-entry">${window.t('waiting_for_extraction')}</div>`);
    }
    
    // Update log container
    logsContainer.innerHTML = allLogs.join('');
    
    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Reset extraction form
export function resetExtractForm() {
    // Reset model configuration
    document.getElementById('extractModelPath').value = '';
    document.getElementById('extractGpuDevices').value = '0';
    
    // Reset method selection
    document.getElementById('extractMethod').value = 'diffmean'; // 将默认方法从lat改为diffmean
    document.getElementById('extractTargetLayer').value = '';
    document.getElementById('extractTokenPos').value = -1; // 使用整数-1作为默认值
    document.getElementById('extractNormalize').checked = true;
    
    // SAE选项已被移除，不再需要重置
    
    // Clear all sample pairs
    const samplePairsList = document.getElementById('samplePairsList');
    samplePairsList.innerHTML = '';
    
    // Add empty message
    const emptyMessage = document.createElement('div');
    emptyMessage.className = 'empty-samples-message';
    emptyMessage.setAttribute('data-i18n', 'no_samples_message');
    emptyMessage.textContent = window.t ? window.t('no_samples_message') : 'No sample pairs added yet. Click "Add Sample Pair" to add your first pair.';
    samplePairsList.appendChild(emptyMessage);
    
    // Reset hidden fields
    document.getElementById('extractPositiveSamples').value = '';
    document.getElementById('extractNegativeSamples').value = '';
    
    // Reset counter
    samplePairsCount = 0;
    
    // Reset output configuration
    document.getElementById('extractOutputPath').value = './extracted_vectors/control_vector.gguf';
    document.getElementById('extractVectorName').value = 'extracted_vector';
    
    // Stop extraction status polling
    stopExtractionStatusPolling();
    
    // Hide results and errors
    document.getElementById('extractResponse').style.display = 'none';
    document.getElementById('extractError').style.display = 'none';
    
    // Update method options to match the selected method
    updateExtractionMethodOptions();
}

// Dynamically load extraction configuration options
export async function loadExtractConfigOptions() {
    try {
        const response = await fetch(window.EasySteerConfig.getApiUrl('/api/extract-configs'));
        if (response.ok) {
            const data = await response.json();
            const extractConfigSelect = document.getElementById('extractConfigSelect');
            
            // Clear existing options (except the default one)
            while (extractConfigSelect.children.length > 1) {
                extractConfigSelect.removeChild(extractConfigSelect.lastChild);
            }
            
            // Add dynamically loaded configuration options
            data.configs.forEach(config => {
                const option = document.createElement('option');
                option.value = config.name;
                option.textContent = config.display_name;
                extractConfigSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.log('Could not load extraction configuration options, using default options');
    }
}

// Import selected extraction configuration
export async function importSelectedExtractConfig() {
    const extractConfigSelect = document.getElementById('extractConfigSelect');
    const selectedConfig = extractConfigSelect.value;
    
    if (!selectedConfig) {
        showExtractError(window.t('error_select_extract_config'));
        return;
    }
    
    try {
        showExtractResponse({ message: window.t('importing_extract_config', { configName: extractConfigSelect.options[extractConfigSelect.selectedIndex].text }) });
        
        // Get extraction config file from the backend
        const response = await fetch(window.EasySteerConfig.getApiUrl(`/api/extract-config/${selectedConfig}`));
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const config = await response.json();
        
        // Handle both nested and flat config structures
        // Set model configuration
        document.getElementById('extractModelPath').value = config.model?.path || config.model_path || '';
        document.getElementById('extractGpuDevices').value = config.model?.gpu_devices || config.gpu_devices || '0';
        
        // Set method configuration
        document.getElementById('extractMethod').value = config.method?.name || config.method || 'diffmean';
        document.getElementById('extractTargetLayer').value = config.method?.target_layer !== undefined ? config.method.target_layer : 
                                                             (config.target_layer !== undefined ? config.target_layer : '');
        
        // 处理token_pos，确保它是一个整数值
        let tokenPos = config.method?.token_pos || config.token_pos || -1;
        if (typeof tokenPos === 'string' && tokenPos !== '-1' && isNaN(parseInt(tokenPos))) {
            // 如果是字符串且不是数字形式，则转换为相应的索引
            if (tokenPos === 'first') tokenPos = 0;
            else tokenPos = -1; // 默认使用最后一个token
        }
        document.getElementById('extractTokenPos').value = parseInt(tokenPos);
        document.getElementById('extractNormalize').checked = config.method?.normalize !== undefined ? config.method.normalize : 
                                                             (config.normalize !== undefined ? config.normalize : true);
        
        // SAE选项已被移除，不再需要设置
        // 我们已经限制了方法只能是DiffMean和PCA，所以不会有SAE配置
        
        // Update method options based on the selected method
        updateExtractionMethodOptions();
        
        // Clear existing sample pairs
        const samplePairsList = document.getElementById('samplePairsList');
        samplePairsList.innerHTML = '';
        samplePairsCount = 0;
        
        // Add imported samples as pairs - handle both nested and flat structures
        const positiveSamples = config.data?.positive_samples || config.positive_samples || [];
        const negativeSamples = config.data?.negative_samples || config.negative_samples || [];
        
        if (positiveSamples.length > 0 || negativeSamples.length > 0) {
            // Create sample pairs by pairing positive and negative samples
            const maxPairs = Math.max(positiveSamples.length, negativeSamples.length);
            for (let i = 0; i < maxPairs; i++) {
                const positive = i < positiveSamples.length ? positiveSamples[i] : '';
                const negative = i < negativeSamples.length ? negativeSamples[i] : '';
                addSamplePair(positive, negative);
            }
        } else {
            // Add empty message if no samples
            const emptyMessage = document.createElement('div');
            emptyMessage.className = 'empty-samples-message';
            emptyMessage.setAttribute('data-i18n', 'no_samples_message');
            emptyMessage.textContent = window.t ? window.t('no_samples_message') : 'No sample pairs added yet. Click "Add Sample Pair" to add your first pair.';
            samplePairsList.appendChild(emptyMessage);
        }
        
        // Set output configuration
        document.getElementById('extractOutputPath').value = config.output?.path || config.output_path || './extracted_vectors/control_vector.safetensors';
        document.getElementById('extractVectorName').value = config.output?.name || config.vector_name || 'extracted_vector';
        
        showExtractResponse({
            message: window.t('extract_import_success_message'),
            description: window.t('extract_import_success_description', { configName: extractConfigSelect.options[extractConfigSelect.selectedIndex].text })
        });
        
        // Clear selection box
        extractConfigSelect.value = '';
        
    } catch (error) {
        showExtractError(window.t('extract_import_fail_error') + ': ' + error.message);
    }
} 

// Initialize extraction interface
export function initExtractionInterface() {
    // Set up initial empty state
    resetExtractForm();
    
    // Expose functions to global scope
    window.addSamplePair = addSamplePair;
    window.removeSamplePair = removeSamplePair;
    window.updateSamplePairs = updateSamplePairs;
} 