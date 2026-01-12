// 训练功能模块

// Training status polling interval ID
let trainingStatusInterval = null;
// Keep track of training examples count
let trainingExamplesCount = 0;

// Display training response
export function showTrainResponse(data) {
    const responseDiv = document.getElementById('trainResponse');
    const responseContent = document.getElementById('trainResponseContent');
    const errorDiv = document.getElementById('trainError');
    
    if (data.success) {
        // If training started successfully, display progress UI and start polling
        responseContent.innerHTML = createTrainingProgressHTML();
        startTrainingStatusPolling();
    } else {
        // Display error message
        responseContent.textContent = JSON.stringify(data, null, 2);
    }
    
    responseDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    
    // Scroll to the result
    responseDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display training error message
export function showTrainError(message) {
    const errorDiv = document.getElementById('trainError');
    const errorContent = document.getElementById('trainErrorContent');
    const responseDiv = document.getElementById('trainResponse');
    
    errorContent.textContent = message;
    errorDiv.style.display = 'block';
    responseDiv.style.display = 'none';
    
    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Add a new training example
export function addTrainingExample(inputText = '', outputText = '') {
    // Get the template
    const template = document.getElementById('trainingExampleTemplate');
    const examplesList = document.getElementById('trainingExamplesList');
    
    // Remove empty message if it exists
    const emptyMessage = examplesList.querySelector('.empty-examples-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    // Clone the template content
    const clone = template.content.cloneNode(true);
    
    // Set unique index and update content
    const exampleItem = clone.querySelector('.training-example-item');
    const index = trainingExamplesCount++;
    exampleItem.dataset.index = index;
    
    // Set the example number
    clone.querySelector('.example-number').textContent = `#${index + 1}`;
    
    // Set remove button action
    const removeBtn = clone.querySelector('.remove-example-btn');
    removeBtn.onclick = function() {
        removeTrainingExample(index);
    };
    
    // Set input and output values if provided
    if (inputText) {
        clone.querySelector('.example-input-text').value = inputText;
    }
    if (outputText) {
        clone.querySelector('.example-output-text').value = outputText;
    }
    
    // Add change event listeners
    const inputTextarea = clone.querySelector('.example-input-text');
    const outputTextarea = clone.querySelector('.example-output-text');
    inputTextarea.onchange = updateTrainingExamples;
    outputTextarea.onchange = updateTrainingExamples;
    
    // Append to the list
    examplesList.appendChild(clone);
    
    // Update the hidden JSON field
    updateTrainingExamples();
    
    // Return the index of the added example
    return index;
}

// Remove a training example
export function removeTrainingExample(index) {
    const examplesList = document.getElementById('trainingExamplesList');
    const example = examplesList.querySelector(`.training-example-item[data-index="${index}"]`);
    
    if (example) {
        example.remove();
        
        // Re-number remaining examples
        const examples = examplesList.querySelectorAll('.training-example-item');
        examples.forEach((example, i) => {
            // Update the display number
            example.querySelector('.example-number').textContent = `#${i + 1}`;
            // Update the data-index attribute
            example.dataset.index = i;
            // Update the remove button onclick handler
            const removeBtn = example.querySelector('.remove-example-btn');
            removeBtn.onclick = function() {
                removeTrainingExample(i);
            };
        });
        
        // Show empty message if no examples left
        if (examples.length === 0) {
            const emptyMessage = document.createElement('div');
            emptyMessage.className = 'empty-examples-message';
            emptyMessage.setAttribute('data-i18n', 'no_examples_message');
            emptyMessage.textContent = window.t ? window.t('no_examples_message') : 'No training examples added yet. Click "Add Example" to add your first example.';
            examplesList.appendChild(emptyMessage);
        }
        
        // Update the trainingExamplesCount to match the current number of examples
        trainingExamplesCount = examples.length;
        
        // Update the hidden JSON field
        updateTrainingExamples();
    }
}

// Update the hidden JSON field with the current examples
export function updateTrainingExamples() {
    const examplesList = document.getElementById('trainingExamplesList');
    const examples = examplesList.querySelectorAll('.training-example-item');
    const trainingExamplesField = document.getElementById('trainingExamples');
    
    const examplesArray = [];
    
    examples.forEach(example => {
        const input = example.querySelector('.example-input-text').value;
        const output = example.querySelector('.example-output-text').value;
        examplesArray.push([input, output]);
    });
    
    // Store as JSON string in hidden field
    trainingExamplesField.value = JSON.stringify(examplesArray);
    
    return examplesArray;
}

// Start training
export async function startTraining() {
    // First update examples to ensure the latest data
    updateTrainingExamples();
    
    const trainConfig = {
        // Model configuration
        model_path: document.getElementById('trainModelPath').value,
        gpu_devices: document.getElementById('trainGpuDevices').value,
        output_dir: document.getElementById('outputDir').value,
        
        // ReFT configuration
        reft_config: {
            layer: parseInt(document.getElementById('trainLayer').value),
            component: document.getElementById('trainComponent').value,
            low_rank_dimension: parseInt(document.getElementById('lowRankDim').value)
        },
        
        // Training parameters
        training_args: {
            num_train_epochs: parseFloat(document.getElementById('numEpochs').value),
            per_device_train_batch_size: parseInt(document.getElementById('batchSize').value),
            learning_rate: parseFloat(document.getElementById('learningRate').value),
            logging_steps: parseInt(document.getElementById('loggingSteps').value),
            save_steps: parseInt(document.getElementById('saveSteps').value)
        },
        
        // Training data from hidden field
        training_examples: document.getElementById('trainingExamples').value
    };

    // Validate required fields
    if (!trainConfig.model_path || !trainConfig.output_dir) {
        showTrainError(window.t('required_fields_error'));
        return;
    }

    // Check if there are any examples
    const examples = updateTrainingExamples();
    if (examples.length === 0) {
        showTrainError(window.t ? window.t('no_examples_error') : 'Please add at least one training example.');
        return;
    }

    // Validate training data format
    try {
        const parsedExamples = JSON.parse(trainConfig.training_examples);
        trainConfig.training_examples = parsedExamples;
    } catch (error) {
        showTrainError(window.t('train_data_format_error') + ': ' + error.message);
        return;
    }

    // Show loading state
    const trainButton = document.querySelector('[onclick="startTraining()"]');
    const originalHTML = trainButton.innerHTML;
    trainButton.innerHTML = '<span class="loading"></span> ' + window.t('training_in_progress');
    trainButton.disabled = true;

    try {
        // Send request to the backend
        const apiUrl = window.EasySteerConfig.getApiUrl('/api/train');
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept-Language': window.currentLanguage
            },
            body: JSON.stringify(trainConfig)
        });

        const data = await response.json();

        if (response.ok) {
            showTrainResponse(data);
        } else {
            showTrainError(data.error || window.t('training_failed_error'));
        }
    } catch (error) {
        console.error('Training error:', error);
        showTrainError(window.t('network_error'));
    } finally {
        // Restore button state
        trainButton.innerHTML = originalHTML;
        trainButton.disabled = false;
    }
}

// Create training progress HTML
function createTrainingProgressHTML() {
    return `
        <div class="logs-container" id="trainingLogs">
            <div class="log-entry status-entry">${window.t('initializing_training')}</div>
        </div>
    `;
}

// Start polling for training status
function startTrainingStatusPolling() {
    // Clear previous polling
    if (trainingStatusInterval) {
        clearInterval(trainingStatusInterval);
    }
    
    // Get status immediately once
    updateTrainingStatus();
    
    // Poll every 2 seconds
    trainingStatusInterval = setInterval(updateTrainingStatus, 2000);
}

// Stop polling for training status
function stopTrainingStatusPolling() {
    if (trainingStatusInterval) {
        clearInterval(trainingStatusInterval);
        trainingStatusInterval = null;
    }
}

// Update training status
async function updateTrainingStatus() {
    try {
        const response = await fetch(window.EasySteerConfig.getApiUrl('/api/train-status'));
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const status = await response.json();
        displayTrainingStatus(status);
        
        // If training is finished, stop polling
        if (!status.is_training) {
            stopTrainingStatusPolling();
        }
        
    } catch (error) {
        console.error('Failed to get training status:', error);
        // Stop polling on network error
        stopTrainingStatusPolling();
    }
}

// Display training status
function displayTrainingStatus(status) {
    const logsContainer = document.getElementById('trainingLogs');
    if (!logsContainer) return;
    
    // Create log content array
    let allLogs = [];
    
    // Add status message as a special log entry
    if (status.status_message) {
        let statusClass = 'status-entry';
        
        if (status.error_message) {
            statusClass += ' error-status';
        } else if (!status.is_training && status.status_message.includes('完成')) {
            statusClass += ' success-status';
        } else if (status.is_training) {
            statusClass += ' training-status';
        }
        
        allLogs.push(`<div class="log-entry ${statusClass}">
            <strong>📊 ${status.status_message}</strong>
        </div>`);
    }
    
    // Add normal training logs
    if (status.logs && status.logs.length > 0) {
        const formattedLogs = status.logs.slice(-20).map(log => {
            // Add colors for different types of information
            let coloredLog = log;
            
            // Timestamp
            coloredLog = coloredLog.replace(/\[\d{2}:\d{2}:\d{2}\]/g, '<span style="color: #6b7280;">$&</span>');
            
            // Epoch info
            coloredLog = coloredLog.replace(/Epoch: [\d.]+/g, '<span style="color: #2563eb; font-weight: 600;">$&</span>');
            
            // Loss info
            coloredLog = coloredLog.replace(/Loss: [\d.]+/g, '<span style="color: #dc2626; font-weight: 600;">$&</span>');
            
            // Gradient info
            coloredLog = coloredLog.replace(/Grad: [\d.]+/g, '<span style="color: #ea580c; font-weight: 600;">$&</span>');
            
            // Learning rate
            coloredLog = coloredLog.replace(/LR: [\d.e-]+/g, '<span style="color: #16a34a; font-weight: 600;">$&</span>');
            
            // Runtime and speed
            coloredLog = coloredLog.replace(/Runtime: [\d.]+s/g, '<span style="color: #7c3aed; font-weight: 500;">$&</span>');
            coloredLog = coloredLog.replace(/Speed: [\d.]+ samples\/s/g, '<span style="color: #c2410c; font-weight: 500;">$&</span>');
            
            return `<div class="log-entry">${coloredLog}</div>`;
        });
        
        allLogs = allLogs.concat(formattedLogs);
    }
    
    // If there are no logs, show waiting message
    if (allLogs.length === 0) {
        allLogs.push(`<div class="log-entry status-entry">${window.t('waiting_for_training')}</div>`);
    }
    
    // Update log container
    logsContainer.innerHTML = allLogs.join('');
    
    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Reset training form
export function resetTrainForm() {
    // Model configuration
    document.getElementById('trainModelPath').value = '';
    document.getElementById('trainGpuDevices').value = '0';
    document.getElementById('outputDir').value = './results/loreft';
    
    // ReFT configuration
    document.getElementById('trainLayer').value = '8';
    document.getElementById('trainComponent').value = 'block_output';
    document.getElementById('lowRankDim').value = '4';
    
    // Training parameters
    document.getElementById('numEpochs').value = '100';
    document.getElementById('batchSize').value = '10';
    document.getElementById('learningRate').value = '0.004';
    document.getElementById('loggingSteps').value = '40';
    document.getElementById('saveSteps').value = '500';
    
    // Clear all training examples
    const examplesList = document.getElementById('trainingExamplesList');
    examplesList.innerHTML = '';
    
    // Add empty message
    const emptyMessage = document.createElement('div');
    emptyMessage.className = 'empty-examples-message';
    emptyMessage.setAttribute('data-i18n', 'no_examples_message');
    emptyMessage.textContent = window.t ? window.t('no_examples_message') : 'No training examples added yet. Click "Add Example" to add your first example.';
    examplesList.appendChild(emptyMessage);
    
    // Reset hidden field
    document.getElementById('trainingExamples').value = '[]';
    
    // Reset counter
    trainingExamplesCount = 0;
    
    // Stop training status polling
    stopTrainingStatusPolling();
    
    // Hide results and errors
    document.getElementById('trainResponse').style.display = 'none';
    document.getElementById('trainError').style.display = 'none';
}

// Dynamically load training configuration options
export async function loadTrainConfigOptions() {
    try {
        const response = await fetch(window.EasySteerConfig.getApiUrl('/api/train-configs'));
        if (response.ok) {
            const data = await response.json();
            const trainConfigSelect = document.getElementById('trainConfigSelect');
            
            // Clear existing options (except the default one)
            while (trainConfigSelect.children.length > 1) {
                trainConfigSelect.removeChild(trainConfigSelect.lastChild);
            }
            
            // Add dynamically loaded configuration options
            data.configs.forEach(config => {
                const option = document.createElement('option');
                option.value = config.name;
                option.textContent = config.display_name;
                trainConfigSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.log('Could not load training configuration options');
    }
}

// Import selected training configuration
export async function importSelectedTrainConfig() {
    const trainConfigSelect = document.getElementById('trainConfigSelect');
    const selectedConfig = trainConfigSelect.value;
    
    if (!selectedConfig) {
        showTrainError(window.t('error_select_train_config'));
        return;
    }
    
    try {
        showTrainResponse({ message: window.t('importing_train_config', { configName: trainConfigSelect.options[trainConfigSelect.selectedIndex].text }) });
        
        // Get training config file from the backend
        const response = await fetch(window.EasySteerConfig.getApiUrl(`/api/train-config/${selectedConfig}`));
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const config = await response.json();
        
        // Set model configuration
        document.getElementById('trainModelPath').value = config.model.path || '';
        document.getElementById('trainGpuDevices').value = config.model.gpu_devices || '0';
        document.getElementById('outputDir').value = config.training.output_dir || './results/loreft';
        
        // Set ReFT configuration
        document.getElementById('trainLayer').value = config.reft.layer || '8';
        document.getElementById('trainComponent').value = config.reft.component || 'block_output';
        document.getElementById('lowRankDim').value = config.reft.low_rank_dimension || '4';
        
        // Set training parameters
        document.getElementById('numEpochs').value = config.training.num_train_epochs || '100';
        document.getElementById('batchSize').value = config.training.per_device_train_batch_size || '10';
        document.getElementById('learningRate').value = config.training.learning_rate || '0.004';
        document.getElementById('loggingSteps').value = config.training.logging_steps || '40';
        document.getElementById('saveSteps').value = config.training.save_steps || '500';
        
        // Clear existing examples
        const examplesList = document.getElementById('trainingExamplesList');
        examplesList.innerHTML = '';
        trainingExamplesCount = 0;
        
        // Add imported examples
        if (config.data && config.data.training_examples && Array.isArray(config.data.training_examples)) {
            config.data.training_examples.forEach(example => {
                if (Array.isArray(example) && example.length === 2) {
                    addTrainingExample(example[0], example[1]);
                }
            });
        } else {
            // Add empty message if no examples
            const emptyMessage = document.createElement('div');
            emptyMessage.className = 'empty-examples-message';
            emptyMessage.setAttribute('data-i18n', 'no_examples_message');
            emptyMessage.textContent = window.t ? window.t('no_examples_message') : 'No training examples added yet. Click "Add Example" to add your first example.';
            examplesList.appendChild(emptyMessage);
        }
        
        showTrainResponse({
            message: window.t('train_import_success_message'),
            description: window.t('train_import_success_description', { configName: trainConfigSelect.options[trainConfigSelect.selectedIndex].text })
        });
        
        // Clear selection box
        trainConfigSelect.value = '';
        
    } catch (error) {
        showTrainError(window.t('train_import_fail_error') + ': ' + error.message);
    }
}

// Initialize training interface
export function initTrainingInterface() {
    // Set up initial empty state
    resetTrainForm();
    
    // Expose functions to global scope
    window.addTrainingExample = addTrainingExample;
    window.removeTrainingExample = removeTrainingExample;
    window.updateTrainingExamples = updateTrainingExamples;
} 