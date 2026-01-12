// SAE Explore Module

// 当前模式（query 或 index）
let currentSaeMode = 'query';

// 添加的向量列表
let addedVectors = [];
let vectorConfigCounter = 0;
let currentVectorTab = null;


// 获取当前语言
const getCurrentLanguage = () => localStorage.getItem('language') || 'en';

// Export functions to be used in main.js
export function initSaeExplore() {
    console.log("SAE Explore tab initialized");
    // Initialize API key input from localStorage
    const storedApiKey = localStorage.getItem('neuronpedia_api_key');
    if (storedApiKey && document.getElementById('saeApiKey')) {
        document.getElementById('saeApiKey').value = storedApiKey;
    }
}

// 切换 SAE 查询模式
export function switchSaeMode(mode) {
    currentSaeMode = mode;
    
    // 更新按钮状态
    document.getElementById('searchByQueryBtn').classList.toggle('active', mode === 'query');
    document.getElementById('searchByIndexBtn').classList.toggle('active', mode === 'index');
    
    // 显示/隐藏相应的表单区域
    document.getElementById('queryModeSection').classList.toggle('active', mode === 'query');
    document.getElementById('indexModeSection').classList.toggle('active', mode === 'index');
    
    // 切换结果容器的显示
    document.getElementById('semanticQueryResults').classList.toggle('active', mode === 'query');
    document.getElementById('featureIndexResults').classList.toggle('active', mode === 'index');
    
    // 清空错误信息
    document.getElementById('queryErrorIndicator').style.display = 'none';
    document.getElementById('indexErrorIndicator').style.display = 'none';
}

export async function searchSaeFeatures() {
    console.log("Searching SAE features...");
    // Get values from the form
    const modelId = document.getElementById('saeModelId').value;
    const saeId = document.getElementById('saeId').value;
    const query = document.getElementById('saeQuery').value;
    const apiKey = document.getElementById('saeApiKey')?.value || '';
    
    // Save API key to localStorage if provided
    if (apiKey) {
        localStorage.setItem('neuronpedia_api_key', apiKey);
    } else {
        showSaeError(t('sae_api_key_required'));
        return;
    }
    
    // Validate inputs
    if (!modelId || !saeId || !query) {
        showSaeError(t('sae_fill_all_required_fields'));
        return;
    }
    
    // Show loading state
    const indicator = document.getElementById('queryLoadingIndicator');
    indicator.style.display = 'flex';
    document.getElementById('saeResultsBody').innerHTML = `<tr><td colspan="4" class="empty-results">${t('sae_searching')}</td></tr>`;
    document.getElementById('queryErrorIndicator').style.display = 'none';
    
    // Call the API
    try {
        // 使用完整的URL地址替代相对路径
        const apiUrl = window.EasySteerConfig.getApiUrl('/api/sae/search');
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_id: modelId,
                sae_id: saeId,
                query: query,
                api_key: apiKey
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displaySearchResults(data.results);
        } else {
            throw new Error(data.error || t('sae_search_failed'));
        }
    } catch (error) {
        showSaeError(error.message);
    } finally {
        indicator.style.display = 'none';
    }
}

// 通过特征索引直接获取详情
export async function getFeatureDetailsByIndex() {
    const modelId = document.getElementById('saeModelId').value;
    const saeId = document.getElementById('saeId').value;
    const featureIndex = document.getElementById('saeFeatureIndex').value;
    
    // Validate inputs
    if (!featureIndex) {
        showSaeError(t('sae_feature_index_required'));
        return;
    }
    
    // Show loading state and clear previous results
    const indicator = document.getElementById('indexLoadingIndicator');
    indicator.style.display = 'flex';
    document.getElementById('featureIndexResultsBody').innerHTML = `<p class="empty-results">${t('sae_searching')}</p>`;
    document.getElementById('indexErrorIndicator').style.display = 'none';
    
    try {
        const apiUrl = window.EasySteerConfig.getApiUrl(`/api/sae/feature/${modelId}/${saeId}/${featureIndex}`);
        const response = await fetch(apiUrl);
        const data = await response.json();
        
        if (response.ok) {
            displayFeatureDetailsInTable(data.feature);
        } else {
            throw new Error(data.error || t('sae_feature_details_failed'));
        }
    } catch (error) {
        showSaeError(error.message);
    } finally {
        indicator.style.display = 'none';
    }
}

function displaySearchResults(results) {
    const resultsTableBody = document.getElementById('saeResultsBody');
    
    if (!results || results.length === 0) {
        resultsTableBody.innerHTML = `<tr><td colspan="5" class="empty-results">${t('sae_no_results')}</td></tr>`;
        return;
    }
    
    // 构建表格显示结果
    let html = '';
    
    results.forEach(feature => {
        html += `
            <tr class="sae-result-row">
                <td>${feature.index}</td>
                <td>
                    <div class="feature-description">${feature.description || t('sae_no_description')}</div>
                </td>
                <td class="similarity-score">${(feature.cosine_similarity || 0).toFixed(3)}</td>
                <td>
                    <div class="button-container">
                        <button class="btn-sm" onclick="getFeatureDetails('${document.getElementById('saeModelId').value}', '${document.getElementById('saeId').value}', ${feature.index})" title="${t('sae_view_details_label')}">
                            <i class="fas fa-info-circle"></i>
                        </button>
                    </div>
                </td>
                <td>
                    <div class="button-container">
                        <button class="btn-sm btn-add-vector" onclick="createVectorFromFeature(${feature.index})" title="${t('sae_add_vector_tooltip')}">
                            <i class="fas fa-plus-circle"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `;
    });
    
    resultsTableBody.innerHTML = html;
}

// 直接添加向量，不弹出对话框
window.createVectorFromFeature = async function(featureIndex) {
    // 如果已经添加过该向量，则不重复添加
    if (addedVectors.some(v => v.featureIndex === featureIndex)) {
        // Find the tab for the existing vector and switch to it
        const existingIndex = addedVectors.findIndex(v => v.featureIndex === featureIndex);
        if (existingIndex !== -1) {
            const configId = `sae-vector-${existingIndex}`;
            switchSaeVectorTab(configId);
        }
        return;
    }
    
    // 默认向量名称和比例
    const vectorName = `feature_${featureIndex}`;
    const vectorScale = 1.0;
    
    try {
        // 调用API创建向量
        const apiUrl = window.EasySteerConfig.getApiUrl('/api/sae/extract-vector');
        
        // 获取相应的按钮并添加加载状态
        const addButton = [...document.querySelectorAll('.btn-add-vector')].find(btn => 
            btn.onclick.toString().includes(`createVectorFromFeature(${featureIndex})`));
        
        if (addButton) {
            const originalHtml = addButton.innerHTML;
            addButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            addButton.disabled = true;
        }
        
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                feature_index: featureIndex,
                vector_name: vectorName,
                scale: vectorScale
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // 获取特征描述（使用原生JS方法）
            let description = '';
            const rows = document.querySelectorAll('tr.sae-result-row');
            for (const row of rows) {
                const idCell = row.querySelector('td:first-child');
                if (idCell && idCell.textContent.trim() === String(featureIndex)) {
                    const descCell = row.querySelector('td:nth-child(2)');
                    if (descCell) {
                        description = descCell.textContent;
                    }
                    break;
                }
            }
            
            // 添加到向量列表，包含完整配置信息
            const vectorData = {
                featureIndex: featureIndex,
                path: `../vectors/${featureIndex}.pt`,
                scale: 500, // 默认值修改为500
                description: description,
                // 初始化其他配置参数
                targetLayers: '31',
                normalize: false,
                algorithm: 'direct',
                prefillTokens: '-1',
                prefillPositions: '',
                generateTokens: '-1'
            };
            
            addedVectors.push(vectorData);
            
            // 更新向量列表显示
            updateSaeVectorsList();

            // 切换到新添加的标签页
            const newConfigId = `sae-vector-${addedVectors.length - 1}`;
            switchSaeVectorTab(newConfigId);

        } else {
            throw new Error(data.error || t('sae_vector_failed'));
        }
    } catch (error) {
        showSaeError(error.message);
    } finally {
        // 恢复按钮状态
        const addButton = [...document.querySelectorAll('.btn-add-vector')].find(btn => 
            btn.onclick.toString().includes(`createVectorFromFeature(${featureIndex})`));
        if (addButton) {
            addButton.innerHTML = '<i class="fas fa-plus-circle"></i>';
            addButton.disabled = false;
        }
    }
};

// 从特征索引输入框中获取索引并添加向量
window.createVectorFromFeatureIndex = async function() {
    // 获取输入框中的特征索引
    const featureIndexInput = document.getElementById('saeFeatureIndex');
    const featureIndex = parseInt(featureIndexInput.value);
    
    // 验证输入
    if (!featureIndex || isNaN(featureIndex)) {
        showSaeError('Please enter a valid feature index');
        return;
    }
    
    // 如果已经添加过该向量，则不重复添加
    if (addedVectors.some(v => v.featureIndex === featureIndex)) {
        // Find the tab for the existing vector and switch to it
        const existingIndex = addedVectors.findIndex(v => v.featureIndex === featureIndex);
        if (existingIndex !== -1) {
            const configId = `sae-vector-${existingIndex}`;
            switchSaeVectorTab(configId);
        }
        return;
    }
    
    // 获取Save按钮并添加加载状态
    const saveButton = document.querySelector('.index-query-row .btn-sm');
    const originalHtml = saveButton.innerHTML;
    saveButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    saveButton.disabled = true;
    
    try {
        // 默认向量名称和比例
        const vectorName = `feature_${featureIndex}`;
        const vectorScale = 1.0;
        
        // 1. 首先调用API生成和保存PT文件
        const extractApiUrl = window.EasySteerConfig.getApiUrl('/api/sae/extract-vector');
        const extractResponse = await fetch(extractApiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                feature_index: featureIndex,
                vector_name: vectorName,
                scale: vectorScale
            })
        });
        
        const extractData = await extractResponse.json();
        
        if (!extractResponse.ok || !extractData.success) {
            throw new Error(extractData.error || 'Failed to extract vector');
        }
        
        // 2. 获取特征详情
        let description = '';
        let featureData = null;
        
        // 获取特征详情的API调用
        const apiUrl = window.EasySteerConfig.getApiUrl(`/api/sae/feature/${document.getElementById('saeModelId').value}/${document.getElementById('saeId').value}/${featureIndex}`);
        
        const response = await fetch(apiUrl);
        const data = await response.json();
        
        if (response.ok) {
            featureData = data.feature;
            description = featureData.explanation || `Feature #${featureIndex}`;
        } else {
            throw new Error(data.error || 'Failed to get feature details');
        }
        
        // 3. 添加到向量列表，包含完整配置信息
        const vectorData = {
            featureIndex: featureIndex,
            path: `../vectors/${featureIndex}.pt`,
            scale: 500, // 默认值为500
            description: description,
            // 初始化其他配置参数
            targetLayers: '31',
            normalize: false,
            algorithm: 'direct',
            prefillTokens: '-1',
            prefillPositions: '',
            generateTokens: '-1'
        };
        
        addedVectors.push(vectorData);

// 更新向量列表显示
        updateSaeVectorsList();

        // 切换到新添加的标签页
        const newConfigId = `sae-vector-${addedVectors.length - 1}`;
        switchSaeVectorTab(newConfigId);
        
        // 显示向量容器
        document.getElementById('sae-vectors-container').style.display = 'block';
        
    } catch (error) {
        showSaeError(error.message);
    } finally {
        // 恢复按钮状态
        saveButton.innerHTML = originalHtml;
        saveButton.disabled = false;
    }
};

// 更新向量列表显示为标签页
function updateSaeVectorsList() {
    const vectorsContainer = document.getElementById('sae-vectors-container');
    const tabsContainer = document.getElementById('saeVectorTabs');
    const contentsContainer = document.getElementById('saeVectorTabContents');

    if (!tabsContainer || !contentsContainer) return;
    
    // 显示或隐藏整个区域
    if (addedVectors.length > 0) {
        vectorsContainer.style.display = 'block';
    } else {
        vectorsContainer.style.display = 'none';
        return;
    }
    
    // 保存当前激活的标签页
    const previouslyActiveTab = currentVectorTab;

    // 清空现有内容
    tabsContainer.innerHTML = '';
    contentsContainer.innerHTML = '';
    
    // 重新创建所有标签页和内容
    addedVectors.forEach((vector, index) => {
        const configId = `sae-vector-${index}`;
        
        // 创建标签页按钮
        const tabButton = document.createElement('button');
        tabButton.id = `tab-${configId}`;
        tabButton.className = 'vector-tab';
        tabButton.innerHTML = `
            Feature #${vector.featureIndex}
            <span class="remove-tab" onclick="event.stopPropagation(); removeSaeVector(${index})">
                        <i class="fas fa-times"></i>
            </span>
        `;
        tabButton.onclick = () => switchSaeVectorTab(configId);
        tabsContainer.appendChild(tabButton);
        
        // 创建标签页内容
        const tabContent = document.createElement('div');
        tabContent.id = `content-${configId}`;
        tabContent.className = 'vector-tab-content';
        // 使用与 multi-vector 一致的 Path 输入框样式
        tabContent.innerHTML = `
            <div class="form-groups-container">
                <div class="vector-description">
                    <b>Description:</b> ${vector.description || 'No description available'}
                </div>
                
                <div class="form-group">
                    <label>Path</label>
                    <div class="file-input-wrapper">
                        <input type="text" id="${configId}-path" value="${vector.path}" placeholder="/path/to/steer_vector.safetensors" oninput="updateSaeVectorParam(${index}, 'path', this.value)">
                        <button class="btn-secondary" onclick="document.getElementById('${configId}-file').click()" title="Select local file">
                            <i class="fas fa-folder-open"></i>
                        </button>
                        <input type="file" id="${configId}-file" style="display: none;" accept=".safetensors,.pt,.bin,.gguf" onchange="updateSaeVectorFilePath(${index})">
                    </div>
                        </div>
                        
                <div class="form-group">
                    <label>Scale Factor</label>
                    <div class="slider-container scale-factor-slider">
                        <input type="range" id="${configId}-scale-slider" min="-2000" max="2000" step="0.1" value="${vector.scale || 500}" oninput="updateSaeVectorParam(${index}, 'scale', this.value, true)">
                        <input type="number" id="${configId}-scale" min="-2000" max="2000" step="0.1" value="${vector.scale || 500}" oninput="updateSaeVectorParam(${index}, 'scale', this.value, false)">
                    </div>
                            </div>
                <div class="form-group">
                    <label>Target Layers</label>
                    <input type="text" id="${configId}-layers" value="${vector.targetLayers || '31'}" placeholder="e.g., 30,31" oninput="updateSaeVectorParam(${index}, 'targetLayers', this.value)">
                        </div>
                <div class="form-group">
                    <label>Algorithm</label>
                    <select id="${configId}-algorithm" onchange="updateSaeVectorParam(${index}, 'algorithm', this.value)">
                        <option value="direct" ${vector.algorithm === 'direct' ? 'selected' : ''}>Direct Algorithm</option>
                        <option value="loreft" ${vector.algorithm === 'loreft' ? 'selected' : ''}>LoReft</option>
                            </select>
                        </div>
                <div class="form-group">
                    <label>Prefill Trigger Token IDs</label>
                    <input type="text" id="${configId}-prefill-tokens" value="${vector.prefillTokens || '-1'}" placeholder="e.g., 100,200 or -1 to apply to all" oninput="updateSaeVectorParam(${index}, 'prefillTokens', this.value)">
                    </div>
                <div class="form-group">
                    <label>Prefill Trigger Positions</label>
                    <input type="text" id="${configId}-prefill-positions" value="${vector.prefillPositions || ''}" placeholder="e.g., -1,-2" oninput="updateSaeVectorParam(${index}, 'prefillPositions', this.value)">
                        </div>
                <div class="form-group">
                    <label>Generate Trigger Token IDs</label>
                    <input type="text" id="${configId}-generate-tokens" value="${vector.generateTokens || ''}" placeholder="e.g., 400,500 or -1 to apply to all" oninput="updateSaeVectorParam(${index}, 'generateTokens', this.value)">
                        </div>
                <div class="form-group checkbox-group">
                    <label>
                        <input type="checkbox" id="${configId}-normalize" ${vector.normalize ? 'checked' : ''} onchange="updateSaeVectorParam(${index}, 'normalize', this.checked)">
                        <span>Normalize Vector</span>
                    </label>
                </div>
            </div>
        `;
        contentsContainer.appendChild(tabContent);
    });

    // 恢复之前的激活状态或默认激活第一个
    if (addedVectors.length > 0) {
        let tabToActivate = previouslyActiveTab;
        // 检查之前的tab是否存在
        if (!document.getElementById(`tab-${tabToActivate}`)) {
            tabToActivate = `sae-vector-${addedVectors.length - 1}`;
        }
        switchSaeVectorTab(tabToActivate);
    } else {
        currentVectorTab = null;
    }
}

// 切换向量配置标签页
function switchSaeVectorTab(configId) {
    document.querySelectorAll('#saeVectorTabs .vector-tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('#saeVectorTabContents .vector-tab-content').forEach(content => content.classList.remove('active'));
    
    const tabElement = document.getElementById(`tab-${configId}`);
    const contentElement = document.getElementById(`content-${configId}`);
    
    if (tabElement) tabElement.classList.add('active');
    if (contentElement) contentElement.classList.add('active');
    
    currentVectorTab = configId;
}

// 更新向量参数
window.updateSaeVectorParam = function(index, param, value, fromSlider = false) {
    if (index < 0 || index >= addedVectors.length) return;
    
    const vector = addedVectors[index];
    const configId = `sae-vector-${index}`;

        if (param === 'scale') {
        vector.scale = parseFloat(value);
        if (fromSlider) {
            document.getElementById(`${configId}-scale`).value = value;
        } else {
            document.getElementById(`${configId}-scale-slider`).value = value;
        }
    } else if (param === 'normalize') {
        vector.normalize = value;
    } else {
        vector[param] = value;
    }
};

// 移除向量
window.removeSaeVector = function(index) {
    if (index < 0 || index >= addedVectors.length) return;
    
    const removingCurrent = (`sae-vector-${index}` === currentVectorTab);
    
        addedVectors.splice(index, 1);
    updateSaeVectorsList();
    
    if (removingCurrent && addedVectors.length > 0) {
        switchSaeVectorTab(`sae-vector-0`);
    }
};

// 清除所有向量
window.clearVectors = function() {
    addedVectors = [];
    currentVectorTab = null;
    updateSaeVectorsList();
};

// 显示生成响应结果
function showSaeGenerationResponse(data) {
    const responseDiv = document.getElementById('saeResponse');
    const responseContent = document.getElementById('saeResponseContent');
    const errorDiv = document.getElementById('saeError');
    
    // 格式化并显示生成结果
    if (data.generated_text) {
        // 检查是否有基准和转向文本进行比较
        if (data.baseline_text) {
            // 创建垂直比较视图，响应堆叠（基准优先）
            const comparisonHTML = `
                <div class="response-comparison-vertical">
                    <div class="response-section baseline-section">
                        <h4><i class="fas fa-robot"></i> Baseline</h4>
                        <pre class="baseline-text">${escapeHtml(data.baseline_text)}</pre>
                    </div>
                    <div class="response-section steered-section">
                        <h4><i class="fas fa-magic"></i> Steered</h4>
                        <pre class="steered-text">${escapeHtml(data.generated_text)}</pre>
                    </div>
                </div>
            `;
            responseContent.innerHTML = comparisonHTML;
        } else {
            // 常规单一响应
            responseContent.innerHTML = `<pre>${escapeHtml(data.generated_text)}</pre>`;
        }
    } else {
        // 非生成响应（消息等）
        responseContent.innerHTML = `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
    }
    
    responseDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    
    // 滚动到结果
    responseDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 显示生成错误消息
function showSaeGenerationError(message) {
    const errorDiv = document.getElementById('saeError');
    const errorContent = document.getElementById('saeErrorContent');
    const responseDiv = document.getElementById('saeResponse');
    
    errorContent.textContent = message;
    errorDiv.style.display = 'block';
    responseDiv.style.display = 'none';
    
    // 滚动到错误
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 应用向量生成文本
window.applyVectors = async function() {
    if (addedVectors.length === 0) {
        showSaeGenerationError('No vectors have been added. Please add at least one vector.');
        return;
    }
    
    // 构建API请求数据
    const config = {
        // 模型配置
        model_path: document.getElementById('saeModelPath').value,
        gpu_devices: document.getElementById('saeGpuDevices').value,
        instruction: document.getElementById('saeInstruction').value,
        
        // 采样参数
        sampling_params: {
            temperature: parseFloat(document.getElementById('saeTemperature').value),
            max_tokens: parseInt(document.getElementById('saeMaxTokens').value),
            repetition_penalty: parseFloat(document.getElementById('saeRepetitionPenalty').value)
        },
        
        // Steer Vector配置
        steer_vector_name: document.getElementById('saeSVName').value,
        steer_vector_int_id: parseInt(document.getElementById('saeSVId').value),
        
        // 冲突解决方法
        conflict_resolution: document.getElementById('saeConflictResolution').value,
        
        // 多向量配置
        vector_configs: addedVectors.map((vector, index) => ({
            path: vector.path,
            scale: parseFloat(vector.scale),
            algorithm: vector.algorithm || 'direct',
            target_layers: vector.targetLayers ? vector.targetLayers.split(',').map(num => parseInt(num.trim())) : [31],
            prefill_trigger_positions: vector.prefillPositions ? vector.prefillPositions.split(',').map(num => parseInt(num.trim())) : [],
            prefill_trigger_tokens: vector.prefillTokens ? vector.prefillTokens.split(',').map(num => parseInt(num.trim())) : [-1],
            generate_trigger_tokens: vector.generateTokens ? vector.generateTokens.split(',').map(num => parseInt(num.trim())) : [-1],
            normalize: vector.normalize || false
        }))
    };
    
    // 验证必填字段
    if (!config.model_path || !config.instruction) {
        showSaeGenerationError('Model Path and Input Instruction are required fields.');
        return;
    }
    
    // 显示加载状态
    const generateButton = document.getElementById('saeApplyVectorsBtn');
    const originalHTML = generateButton.innerHTML;
    generateButton.innerHTML = '<span class="loading"></span> Generating...';
    generateButton.disabled = true;
    
    try {
        // 发送请求到后端API
        const apiUrl = window.EasySteerConfig.getApiUrl('/api/generate-multi');
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        const data = await response.json();
        
        if (response.ok) {
            showSaeGenerationResponse(data);
        } else {
            showSaeGenerationError(data.error || 'Failed to generate response.');
        }
    } catch (error) {
        console.error('Error during generation:', error);
        showSaeGenerationError(`Network error: ${error.message}`);
    } finally {
        // 恢复按钮状态
        generateButton.innerHTML = originalHTML;
        generateButton.disabled = false;
    }
};

// HTML转义函数
function escapeHtml(text) {
    if (!text) return '';
    
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 新函数：在表格中显示特征详情
function displayFeatureDetailsInTable(feature) {
    const resultsBody = document.getElementById('featureIndexResultsBody');
    if (!feature) {
        resultsBody.innerHTML = `<p class="empty-results" data-i18n="sae_no_results">${t('sae_no_results')}</p>`;
        return;
    }

    let html = '<table>';
    
    // Basic Info
    html += `<tr><td class="detail-label">${t('sae_model_label')}</td><td class="detail-value">${feature.basic_info?.modelId || t('sae_unknown')}</td></tr>`;
    html += `<tr><td class="detail-label">${t('sae_layer_label')}</td><td class="detail-value">${feature.basic_info?.layer || t('sae_unknown')}</td></tr>`;
    html += `<tr><td class="detail-label">${t('sae_index_label')}</td><td class="detail-value">${feature.basic_info?.index || t('sae_unknown')}</td></tr>`;
    html += `<tr><td class="detail-label">${t('sae_description_label')}</td><td class="detail-value">${feature.explanation || t('sae_no_description')}</td></tr>`;
    if (feature.sparsity) {
        html += `<tr><td class="detail-label">${t('sae_sparsity_label')}</td><td class="detail-value">${(feature.sparsity * 100).toFixed(2)}%</td></tr>`;
    }

    // Top Activating Tokens
    if (feature.top_activating_tokens && feature.top_activating_tokens.length > 0) {
        html += `<tr><td class="detail-label section-header">${t('sae_top_activating_tokens')}</td><td class="detail-value"><div class="token-list">${feature.top_activating_tokens.map(t => `<div class="token-item">${t.token} <span class="value">${t.activation_value.toFixed(3)}</span></div>`).join('')}</div></td></tr>`;
    }

    // Top Inhibiting Tokens
    if (feature.top_inhibiting_tokens && feature.top_inhibiting_tokens.length > 0) {
        html += `<tr><td class="detail-label section-header">${t('sae_top_inhibiting_tokens')}</td><td class="detail-value"><div class="token-list">${feature.top_inhibiting_tokens.map(t => `<div class="token-item">${t.token} <span class="value">${t.activation_value.toFixed(3)}</span></div>`).join('')}</div></td></tr>`;
    }
    
    // Activation Example
    if (feature.activation_example) {
        html += `<tr><td class="detail-label section-header">${t('sae_activation_example')}</td><td class="detail-value">` +
                `<div class="detail-item nested"><span class="nested-label">${t('sae_max_value_label')}:</span><span>${feature.activation_example.max_value.toFixed(3)}</span></div>` +
                `<div class="detail-item nested"><span class="nested-label">${t('sae_trigger_token_label')}:</span><span>${feature.activation_example.trigger_token}</span></div>` +
                `<div class="detail-item nested"><span class="nested-label">${t('sae_context_label')}:</span><span class="context-text">${feature.activation_example.context}</span></div>` +
                `</td></tr>`;
    }

    html += '</table>';
    resultsBody.innerHTML = html;
}


// This will be called from onclick
window.getFeatureDetails = async function(modelId, saeId, featureIndex) {
    const viewDetailsButton = event.target.closest('button');
    if (viewDetailsButton) {
        viewDetailsButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        viewDetailsButton.disabled = true;
    }
    
    try {
        // 使用完整的URL地址替代相对路径
        const apiUrl = window.EasySteerConfig.getApiUrl(`/api/sae/feature/${modelId}/${saeId}/${featureIndex}`);
        const response = await fetch(apiUrl);
        const data = await response.json();
        
        if (response.ok) {
            displayFeatureDetails(data.feature);
        } else {
            throw new Error(data.error || t('sae_feature_details_failed'));
        }
    } catch (error) {
        showSaeError(error.message);
    } finally {
        if (viewDetailsButton) {
            viewDetailsButton.innerHTML = '<i class="fas fa-info-circle"></i>';
            viewDetailsButton.disabled = false;
        }
    }
};

function displayFeatureDetails(feature) {
    // Make sure we're using the global t function and getting a fresh translation
    // This ensures we show the correct language in the modal title
    const index = feature.basic_info?.index || '';
    const title = window.t ? window.t('sae_feature_details') : '特征详情';
    
    // Get translation function
    const translate = window.t || (key => key);
    
    // 创建详细信息模态框
    const modalHtml = `
        <div class="feature-detail-modal">
            <div class="feature-detail-header" style="display: flex; justify-content: space-between; align-items: center;">
                <h3>${title} #${index}</h3>
                <button class="close-btn" style="background: none; border: none; color: white; font-size: 20px; cursor: pointer;" onclick="document.querySelector('.feature-detail-modal-container').remove()">&times;</button>
            </div>
            <div class="feature-detail-content">
                <div class="feature-detail-section">
                    <h4>${translate('sae_basic_info')}</h4>
                    <div class="detail-item">
                        <span class="detail-label">${translate('sae_model_label')}:</span> 
                        <span class="detail-value">${feature.basic_info?.modelId || translate('sae_unknown')}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">${translate('sae_layer_label')}:</span> 
                        <span class="detail-value">${feature.basic_info?.layer || translate('sae_unknown')}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">${translate('sae_index_label')}:</span> 
                        <span class="detail-value">${feature.basic_info?.index || translate('sae_unknown')}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">${translate('sae_description_label')}:</span> 
                        <span class="detail-value">${feature.explanation || translate('sae_no_description')}</span>
                    </div>
                    ${feature.sparsity ? 
                        `<div class="detail-item">
                            <span class="detail-label">${translate('sae_sparsity_label')}:</span> 
                            <span class="detail-value">${(feature.sparsity * 100).toFixed(2)}%</span>
                        </div>` : 
                        ''}
                </div>
                ${feature.top_activating_tokens && feature.top_activating_tokens.length > 0 ? `
                <div class="feature-detail-section">
                    <h4>${translate('sae_top_activating_tokens')}</h4>
                    <div class="token-list">
                        ${feature.top_activating_tokens.map(token => 
                            `<div class="token-item">${token.token} <span class="value">${token.activation_value.toFixed(3)}</span></div>`
                        ).join('')}
                    </div>
                </div>` : ''}
                ${feature.top_inhibiting_tokens && feature.top_inhibiting_tokens.length > 0 ? `
                <div class="feature-detail-section">
                    <h4>${translate('sae_top_inhibiting_tokens')}</h4>
                    <div class="token-list">
                        ${feature.top_inhibiting_tokens.map(token => 
                            `<div class="token-item">${token.token} <span class="value">${token.activation_value.toFixed(3)}</span></div>`
                        ).join('')}
                    </div>
                </div>` : ''}
                ${feature.activation_example ? `
                <div class="feature-detail-section">
                    <h4>${translate('sae_activation_example')}</h4>
                    <div class="detail-item">
                        <span class="detail-label">${translate('sae_max_value_label')}:</span> 
                        <span class="detail-value">${feature.activation_example.max_value.toFixed(3)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">${translate('sae_trigger_token_label')}:</span> 
                        <span class="detail-value">${feature.activation_example.trigger_token}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">${translate('sae_context_label')}:</span> 
                        <span class="detail-value context-text">${feature.activation_example.context}</span>
                    </div>
                </div>` : ''}
            </div>
            <div class="feature-detail-footer">
                <button class="btn-secondary" onclick="document.querySelector('.feature-detail-modal-container').remove()">${translate('close_btn')}</button>
            </div>
        </div>
    `;

    // 创建遮罩容器
    const modalContainer = document.createElement('div');
    modalContainer.className = 'feature-detail-modal-container';
    document.body.appendChild(modalContainer);
    modalContainer.innerHTML = modalHtml;

    // 点击背景关闭
    modalContainer.addEventListener('click', function(e) {
        if (e.target === modalContainer) {
            modalContainer.remove();
        }
    });
}

export function resetSaeForm() {
    console.log("Resetting SAE form...");
    
    // 重置查询输入
    document.getElementById('saeQuery').value = '';
    document.getElementById('saeFeatureIndex').value = '';
    
    // 隐藏结果和错误
    document.getElementById('saeResultsBody').innerHTML = `<tr><td colspan="4" class="empty-results" data-i18n="sae_enter_query_prompt">${t('sae_enter_query_prompt')}</td></tr>`;
    document.getElementById('featureIndexResultsBody').innerHTML = `<p class="empty-results" data-i18n="sae_enter_index_prompt">${t('sae_enter_index_prompt')}</p>`;
    document.getElementById('queryErrorIndicator').style.display = 'none';
    document.getElementById('indexErrorIndicator').style.display = 'none';
}

export function showSaeError(message) {
    // Hide all loading indicators
    document.getElementById('queryLoadingIndicator').style.display = 'none';
    document.getElementById('indexLoadingIndicator').style.display = 'none';

    // Show the error in the currently active mode
    if (currentSaeMode === 'query') {
        document.getElementById('queryErrorContent').textContent = message;
        document.getElementById('queryErrorIndicator').style.display = 'flex';
    } else {
        document.getElementById('indexErrorContent').textContent = message;
        document.getElementById('indexErrorIndicator').style.display = 'flex';
    }
} 

document.addEventListener('DOMContentLoaded', function() {
    console.log("SAE Explore tab initialized");
    
    // 初始化表格内容
    if (document.getElementById('saeResultsBody')) {
        document.getElementById('saeResultsBody').innerHTML = `<tr><td colspan="4" class="empty-results">${t('sae_enter_query_prompt')}</td></tr>`;
    }
    if (document.getElementById('featureIndexResultsBody')) {
        document.getElementById('featureIndexResultsBody').innerHTML = `<p class="empty-results">${t('sae_enter_index_prompt')}</p>`;
    }
});

// 确保语言切换时也更新动态内容
window.addEventListener('languageChanged', function(e) {
    // 更新动态表格内容
    if (!document.querySelector('#saeResultsBody tr.sae-result-row')) {
        document.getElementById('saeResultsBody').innerHTML = `<tr><td colspan="4" class="empty-results">${t('sae_enter_query_prompt')}</td></tr>`;
    }
    if (!document.querySelector('#featureIndexResultsBody table')) {
        document.getElementById('featureIndexResultsBody').innerHTML = `<p class="empty-results">${t('sae_enter_index_prompt')}</p>`;
    }
}); 

// 更新向量文件路径
window.updateSaeVectorFilePath = function(index) {
    if (index < 0 || index >= addedVectors.length) return;
    
    const configId = `sae-vector-${index}`;
    const fileInput = document.getElementById(`${configId}-file`);
    
    if (fileInput && fileInput.files && fileInput.files[0]) {
        const fileName = fileInput.files[0].name;
        document.getElementById(`${configId}-path`).value = fileName;
        updateSaeVectorParam(index, 'path', fileName);
    }
}; 