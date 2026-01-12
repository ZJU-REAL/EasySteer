// 多向量控制模块

import { parseListInput, escapeHtml } from './utils.js';

// 多向量控制相关变量
export var vectorConfigs = [];
export var vectorConfigCounter = 0;
export var currentVectorTab = null;

// 添加向量配置
export function addVectorConfig() {
    vectorConfigCounter++;
    const configId = `vector-config-${vectorConfigCounter}`;
    const displayNumber = vectorConfigs.length + 1; // 使用当前向量数量+1作为显示编号
    
    const config = {
        id: configId,
        path: '',
        scale: 1.0,
        algorithm: 'direct',
        target_layers: [],
        prefill_trigger_positions: [-1],
        prefill_trigger_tokens: [],
        generate_trigger_tokens: [],
        normalize: false
    };
    
    vectorConfigs.push(config);
    
    // 创建标签页按钮
    const tabButton = document.createElement('button');
    tabButton.id = `tab-${configId}`;
    tabButton.className = 'vector-tab';
    tabButton.innerHTML = `
        Vector ${displayNumber}
        <span class="remove-tab" onclick="event.stopPropagation(); removeVectorConfig('${configId}')">
            <i class="fas fa-times"></i>
        </span>
    `;
    tabButton.onclick = () => switchVectorTab(configId);
    document.getElementById('vectorTabs').appendChild(tabButton);
    
    // 创建标签页内容
    const tabContent = document.createElement('div');
    tabContent.id = `content-${configId}`;
    tabContent.className = 'vector-tab-content';
    tabContent.innerHTML = `
        <div class="form-groups-container">
            <div class="form-group">
                <label>Path</label>
                <div class="file-input-wrapper">
                    <input type="text" id="${configId}-path" placeholder="/path/to/steer_vector.gguf">
                    <button class="btn-secondary" onclick="document.getElementById('${configId}-file').click()">
                        <i class="fas fa-folder-open"></i>
                    </button>
                    <input type="file" id="${configId}-file" style="display: none;" accept=".safetensors,.pt,.bin,.gguf" onchange="updateFilePath('${configId}')">
                </div>
            </div>
            <div class="form-group">
                <label>Scale Factor</label>
                <div class="slider-container">
                    <input type="range" id="${configId}-scale-slider" min="-5" max="5" step="0.1" value="1.0" onchange="updateScale('${configId}')">
                    <input type="number" id="${configId}-scale" min="-5" max="5" step="0.1" value="1.0" onchange="updateScaleSlider('${configId}')">
                </div>
            </div>
            <div class="form-group">
                <label>Algorithm</label>
                <select id="${configId}-algorithm" onchange="updateAlgorithm('${configId}')">
                    <option value="direct">Direct Algorithm</option>
                    <option value="loreft">LoReft</option>
                </select>
            </div>
            <div class="form-group">
                <label>Target Layers</label>
                <input type="text" id="${configId}-layers" placeholder="e.g., 0,1,2,3 or leave empty" onchange="updateLayers('${configId}')">
            </div>
            
            <div class="form-group">
                <label>Prefill Trigger Token IDs</label>
                <input type="text" id="${configId}-prefill-tokens" placeholder="e.g., 100,200,300 or -1 to apply to all" onchange="updatePrefillTokens('${configId}')">
            </div>
            
            <div class="form-group">
                <label>Prefill Trigger Positions</label>
                <input type="text" id="${configId}-prefill-positions" placeholder="e.g., -1,-2,-3,-4" value="-1" onchange="updatePrefillPositions('${configId}')">
            </div>
            
            <div class="form-group">
                <label>Generate Trigger Token IDs</label>
                <input type="text" id="${configId}-generate-tokens" placeholder="e.g., 400,500,600 or -1 to apply to all" onchange="updateGenerateTokens('${configId}')">
            </div>
            
            <div class="form-group checkbox-group">
                <label>
                    <input type="checkbox" id="${configId}-normalize" onchange="updateNormalize('${configId}')">
                    <span>Normalize Vector</span>
                </label>
            </div>
        </div>
    `;
    document.getElementById('vectorTabContents').appendChild(tabContent);
    
    // 切换到新添加的标签页
    switchVectorTab(configId);
    
    // 更新向量数量显示
    updateVectorCount();
    
    return configId;
}

// 切换向量配置标签页
export function switchVectorTab(configId) {
    // 取消所有标签页的激活状态
    document.querySelectorAll('.vector-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.vector-tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // 激活选定的标签页
    document.getElementById(`tab-${configId}`).classList.add('active');
    document.getElementById(`content-${configId}`).classList.add('active');
    
    // 更新当前标签页
    currentVectorTab = configId;
}

// 更新向量配置数量显示和标签
export function updateVectorCount() {
    const countElement = document.querySelector('[data-i18n="add_vector_btn"]');
    if (countElement) {
        countElement.textContent = `Add Vector Configuration (${vectorConfigs.length})`;
    }
    
    // 如果没有向量配置，隐藏标签页区域
    const vectorTabs = document.getElementById('vectorTabs');
    if (vectorTabs) {
        vectorTabs.style.display = vectorConfigs.length > 0 ? 'flex' : 'none';
    }
    
    // 重新编号所有向量标签，确保序号连续
    vectorConfigs.forEach((config, index) => {
        const tabElement = document.getElementById(`tab-${config.id}`);
        if (tabElement) {
            // 更新标签页标题
            const tabTitle = tabElement.textContent.split(/\s+/)[0]; // 保留第一个单词 "Vector"
            tabElement.childNodes[0].nodeValue = `Vector ${index + 1} `; // 设置新的编号
        }
    });
}

// 更新文件路径
export function updateFilePath(configId) {
    const fileInput = document.getElementById(`${configId}-file`);
    const pathInput = document.getElementById(`${configId}-path`);
    
    if (fileInput.files.length > 0) {
        pathInput.value = fileInput.files[0].name;
        
        // 更新配置
        const configIndex = vectorConfigs.findIndex(config => config.id === configId);
        if (configIndex !== -1) {
            vectorConfigs[configIndex].path = pathInput.value;
        }
    }
}

// 更新比例滑块
export function updateScale(configId) {
    const slider = document.getElementById(`${configId}-scale-slider`);
    const input = document.getElementById(`${configId}-scale`);
    input.value = slider.value;
    
    // 更新配置
    const configIndex = vectorConfigs.findIndex(config => config.id === configId);
    if (configIndex !== -1) {
        vectorConfigs[configIndex].scale = parseFloat(slider.value);
    }
}

// 更新比例输入框
export function updateScaleSlider(configId) {
    const slider = document.getElementById(`${configId}-scale-slider`);
    const input = document.getElementById(`${configId}-scale`);
    slider.value = input.value;
    
    // 更新配置
    const configIndex = vectorConfigs.findIndex(config => config.id === configId);
    if (configIndex !== -1) {
        vectorConfigs[configIndex].scale = parseFloat(input.value);
    }
}

// 更新算法
export function updateAlgorithm(configId) {
    const select = document.getElementById(`${configId}-algorithm`);
    
    // 更新配置
    const configIndex = vectorConfigs.findIndex(config => config.id === configId);
    if (configIndex !== -1) {
        vectorConfigs[configIndex].algorithm = select.value;
    }
}

// 更新目标层
export function updateLayers(configId) {
    const input = document.getElementById(`${configId}-layers`);
    
    // 更新配置
    const configIndex = vectorConfigs.findIndex(config => config.id === configId);
    if (configIndex !== -1) {
        vectorConfigs[configIndex].target_layers = parseListInput(input.value);
    }
}

// 更新触发位置
export function updatePositions(configId) {
    const input = document.getElementById(`${configId}-positions`);
    
    // 更新配置
    const configIndex = vectorConfigs.findIndex(config => config.id === configId);
    if (configIndex !== -1) {
        vectorConfigs[configIndex].prefill_trigger_positions = parseListInput(input.value);
    }
}

// 更新触发令牌函数
export function updatePrefillTokens(configId) {
    const input = document.getElementById(`${configId}-prefill-tokens`);
    
    // 更新配置
    const configIndex = vectorConfigs.findIndex(config => config.id === configId);
    if (configIndex !== -1) {
        vectorConfigs[configIndex].prefill_trigger_tokens = parseListInput(input.value);
    }
}

// 更新触发位置函数
export function updatePrefillPositions(configId) {
    const input = document.getElementById(`${configId}-prefill-positions`);
    
    // 更新配置
    const configIndex = vectorConfigs.findIndex(config => config.id === configId);
    if (configIndex !== -1) {
        vectorConfigs[configIndex].prefill_trigger_positions = parseListInput(input.value);
    }
}

// 更新生成触发令牌函数
export function updateGenerateTokens(configId) {
    const input = document.getElementById(`${configId}-generate-tokens`);
    
    // 更新配置
    const configIndex = vectorConfigs.findIndex(config => config.id === configId);
    if (configIndex !== -1) {
        vectorConfigs[configIndex].generate_trigger_tokens = parseListInput(input.value);
    }
}

// 更新归一化选项
export function updateNormalize(configId) {
    const checkbox = document.getElementById(`${configId}-normalize`);
    
    // 更新配置
    const configIndex = vectorConfigs.findIndex(config => config.id === configId);
    if (configIndex !== -1) {
        vectorConfigs[configIndex].normalize = checkbox.checked;
    }
}

// 移除向量配置
export function removeVectorConfig(configId) {
    // 从DOM中移除标签页
    const tabElement = document.getElementById(`tab-${configId}`);
    const contentElement = document.getElementById(`content-${configId}`);
    if (tabElement) tabElement.remove();
    if (contentElement) contentElement.remove();
    
    // 从配置数组中移除
    const configIndex = vectorConfigs.findIndex(config => config.id === configId);
    if (configIndex !== -1) {
        vectorConfigs.splice(configIndex, 1);
    }
    
    // 更新向量数量显示并重新编号
    updateVectorCount();
    
    // 如果删除的是当前激活的标签页，切换到第一个标签页
    if (currentVectorTab === configId) {
        if (vectorConfigs.length > 0) {
            switchVectorTab(vectorConfigs[0].id);
        } else {
            currentVectorTab = null;
        }
    }
}

// 编辑向量配置
export function editVectorConfig(configId) {
    // 当前简单实现，直接聚焦到路径输入框
    const pathInput = document.getElementById(`${configId}-path`);
    if (pathInput) {
        pathInput.focus();
    }
}

// 显示多向量响应
export function showMultiResponse(data) {
    const responseDiv = document.getElementById('multiResponse');
    const responseContent = document.getElementById('multiResponseContent');
    const errorDiv = document.getElementById('multiError');
    
    // 格式化并显示生成结果
    if (data.generated_text) {
        // 检查是否有基准和转向文本进行比较
        if (data.baseline_text) {
            // 创建垂直比较视图，响应堆叠（基准优先）
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

// 显示多向量错误消息
export function showMultiError(message) {
    const errorDiv = document.getElementById('multiError');
    const errorContent = document.getElementById('multiErrorContent');
    const responseDiv = document.getElementById('multiResponse');
    
    errorContent.textContent = message;
    errorDiv.style.display = 'block';
    responseDiv.style.display = 'none';
    
    // 滚动到错误
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 提交多向量配置
export async function submitMultiConfiguration() {
    // 检查是否添加了向量配置
    if (vectorConfigs.length === 0) {
        showMultiError(window.t('no_vector_configs_error'));
        return;
    }
    
    // 基本配置
    const config = {
        // 模型配置
        model_path: document.getElementById('multiModelPath').value,
        gpu_devices: document.getElementById('multiGpuDevices').value,
        instruction: document.getElementById('multiInstruction').value,
        
        // 采样参数
        sampling_params: {
            temperature: parseFloat(document.getElementById('multiTemperature').value),
            max_tokens: parseInt(document.getElementById('multiMaxTokens').value),
            repetition_penalty: parseFloat(document.getElementById('multiRepetitionPenalty').value)
        },
        
        // Steer Vector请求配置
        steer_vector_name: document.getElementById('multiSteerVectorName').value,
        steer_vector_int_id: parseInt(document.getElementById('multiSteerVectorId').value),
        
        // 冲突解决方法
        conflict_resolution: document.getElementById('conflictResolution').value,
        
        // 多向量配置
        vector_configs: vectorConfigs.map(config => ({
            path: config.path,
            scale: config.scale,
            algorithm: config.algorithm,
            target_layers: config.target_layers,
            prefill_trigger_positions: config.prefill_trigger_positions,
            prefill_trigger_tokens: config.prefill_trigger_tokens,
            generate_trigger_tokens: config.generate_trigger_tokens,
            normalize: config.normalize
        }))
    };

    // 验证必填字段
    if (!config.steer_vector_name || !config.steer_vector_int_id || !config.model_path || !config.instruction) {
        showMultiError(window.t('required_fields_error'));
        return;
    }

    // 显示加载状态
    const submitButton = document.querySelector('#multi-vector-mode .btn-primary');
    const originalHTML = submitButton.innerHTML;
    submitButton.innerHTML = '<span class="loading"></span> ' + window.t('generating');
    submitButton.disabled = true;

    try {
        // 发送请求到后端
        const apiUrl = `${window.location.protocol}//${window.location.hostname}:5000/api/generate-multi`;
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
            showMultiResponse(data);
        } else {
            showMultiError(data.error || window.t('submit_failed'));
        }
    } catch (error) {
        console.error('Error:', error);
        showMultiError(window.t('network_error'));
    } finally {
        // 恢复按钮状态
        submitButton.innerHTML = originalHTML;
        submitButton.disabled = false;
    }
}

// 重置多向量表单
export function resetMultiForm() {
    // 模型配置
    document.getElementById('multiModelPath').value = '';
    document.getElementById('multiGpuDevices').value = '0';
    document.getElementById('multiInstruction').value = '';
    
    // 采样参数
    document.getElementById('multiTemperature').value = '0.0';
    document.getElementById('multiTemperatureSlider').value = '0.0';
    document.getElementById('multiMaxTokens').value = '128';
    document.getElementById('multiRepetitionPenalty').value = '1.1';
    document.getElementById('multiRepetitionPenaltySlider').value = '1.1';
    
    // Steer Vector配置
    document.getElementById('multiSteerVectorName').value = '';
    document.getElementById('multiSteerVectorId').value = '';
    document.getElementById('conflictResolution').value = 'sequential';
    
    // 清空向量配置
    document.getElementById('vectorTabs').innerHTML = '';
    document.getElementById('vectorTabContents').innerHTML = '';
    vectorConfigs = [];
    vectorConfigCounter = 0;
    currentVectorTab = null;
    updateVectorCount();
    
    // 隐藏结果和错误
    document.getElementById('multiResponse').style.display = 'none';
    document.getElementById('multiError').style.display = 'none';
}

// 加载多向量配置选项
export async function loadMultiConfigOptions() {
    try {
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/configs`);
        if (response.ok) {
            const data = await response.json();
            const configSelect = document.getElementById('multiConfigSelect');
            
            // 清除现有选项（除默认选项外）
            while (configSelect.children.length > 1) {
                configSelect.removeChild(configSelect.lastChild);
            }
            
            // 添加动态加载的配置选项（仅添加多向量配置）
            data.configs
                .filter(config => config.type === 'multi_vector')
                .forEach(config => {
                    const option = document.createElement('option');
                    option.value = config.name;
                    option.textContent = config.display_name;
                    configSelect.appendChild(option);
                });
        }
    } catch (error) {
        console.log('Could not load multi-vector configuration options, using default options');
    }
}

// 导入选定的多向量配置
export async function importSelectedMultiConfig() {
    const configSelect = document.getElementById('multiConfigSelect');
    const selectedConfig = configSelect.value;
    
    if (!selectedConfig) {
        showMultiError(window.t('error_select_config'));
        return;
    }
    
    try {
        showMultiResponse({ message: window.t('importing_config', { configName: configSelect.options[configSelect.selectedIndex].text }) });
        
        // 从后端获取配置文件
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/config/${selectedConfig}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const config = await response.json();
        
        // 设置模型配置
        document.getElementById('multiModelPath').value = config.model?.path || '';
        document.getElementById('multiGpuDevices').value = config.model?.gpu_devices || '';
        document.getElementById('multiInstruction').value = config.model?.instruction || '';
        
        // 设置采样参数
        document.getElementById('multiTemperature').value = config.sampling?.temperature || '0.0';
        document.getElementById('multiTemperatureSlider').value = config.sampling?.temperature || '0.0';
        document.getElementById('multiMaxTokens').value = config.sampling?.max_tokens || '128';
        document.getElementById('multiRepetitionPenalty').value = config.sampling?.repetition_penalty || '1.1';
        document.getElementById('multiRepetitionPenaltySlider').value = config.sampling?.repetition_penalty || '1.1';
        
        // 设置Steer Vector配置
        document.getElementById('multiSteerVectorName').value = config.steer_vector?.name || '';
        document.getElementById('multiSteerVectorId').value = config.steer_vector?.id || '';
        document.getElementById('conflictResolution').value = config.steer_vector?.conflict_resolution || 'sequential';
        
        // 清空并创建向量配置
        document.getElementById('vectorTabs').innerHTML = '';
        document.getElementById('vectorTabContents').innerHTML = '';
        vectorConfigs = [];
        vectorConfigCounter = 0;
        currentVectorTab = null;
        
        if (config.vector_configs && Array.isArray(config.vector_configs)) {
            // 多向量配置: 遍历每个向量配置并添加
            for (const vecConfig of config.vector_configs) {
                const configId = addVectorConfig();
                const currentConfig = vectorConfigs[vectorConfigs.length - 1];
                
                // 设置基本配置
                document.getElementById(`${configId}-path`).value = vecConfig.path || '';
                document.getElementById(`${configId}-scale`).value = vecConfig.scale || '1.0';
                document.getElementById(`${configId}-scale-slider`).value = vecConfig.scale || '1.0';
                document.getElementById(`${configId}-algorithm`).value = vecConfig.algorithm || 'direct';
                document.getElementById(`${configId}-layers`).value = vecConfig.target_layers || '';
                document.getElementById(`${configId}-normalize`).checked = vecConfig.normalize || false;
                
                // 设置触发器
                document.getElementById(`${configId}-prefill-positions`).value = vecConfig.prefill_trigger_positions || '';
                document.getElementById(`${configId}-prefill-tokens`).value = vecConfig.prefill_trigger_tokens || '';
                document.getElementById(`${configId}-generate-tokens`).value = vecConfig.generate_trigger_tokens || '';
                
                // 更新配置对象
                currentConfig.path = vecConfig.path || '';
                currentConfig.scale = parseFloat(vecConfig.scale || '1.0');
                currentConfig.algorithm = vecConfig.algorithm || 'direct';
                currentConfig.target_layers = parseListInput(vecConfig.target_layers || '');
                currentConfig.prefill_trigger_positions = parseListInput(vecConfig.prefill_trigger_positions || '-1');
                currentConfig.prefill_trigger_tokens = parseListInput(vecConfig.prefill_trigger_tokens || '');
                currentConfig.generate_trigger_tokens = parseListInput(vecConfig.generate_trigger_tokens || '');
                currentConfig.normalize = vecConfig.normalize || false;
            }
            
            // 确保所有向量的编号连续
            updateVectorCount();
            
            // 如果有向量配置，切换到第一个标签页
            if (vectorConfigs.length > 0) {
                switchVectorTab(vectorConfigs[0].id);
            }
        } else {
            // 单向量配置: 将其转换为多向量配置
            const configId = addVectorConfig();
            const vConfig = vectorConfigs[0];
            
            document.getElementById(`${configId}-path`).value = config.steer_vector?.path || '';
            document.getElementById(`${configId}-scale`).value = config.steer_vector?.scale || '1.0';
            document.getElementById(`${configId}-scale-slider`).value = config.steer_vector?.scale || '1.0';
            document.getElementById(`${configId}-algorithm`).value = config.steer_vector?.algorithm || 'direct';
            document.getElementById(`${configId}-layers`).value = config.steer_vector?.target_layers || '';
            document.getElementById(`${configId}-normalize`).checked = config.steer_vector?.normalize || false;
            
            // 设置触发器配置
            if (config.steer_vector?.algorithm === 'loreft') {
                document.getElementById(`${configId}-prefill-positions`).value = config.steer_vector?.prefill_positions || '-1';
            } else {
                document.getElementById(`${configId}-prefill-positions`).value = config.steer_vector?.prefill_trigger_positions || '-1';
                document.getElementById(`${configId}-prefill-tokens`).value = config.steer_vector?.prefill_trigger_tokens || '';
                document.getElementById(`${configId}-generate-tokens`).value = config.steer_vector?.generate_trigger_tokens || '';
            }
            
            // 更新配置对象
            vConfig.path = config.steer_vector?.path || '';
            vConfig.scale = parseFloat(config.steer_vector?.scale || '1.0');
            vConfig.algorithm = config.steer_vector?.algorithm || 'direct';
            vConfig.target_layers = parseListInput(config.steer_vector?.target_layers || '');
            vConfig.prefill_trigger_positions = parseListInput(config.steer_vector?.prefill_positions || 
                                                         config.steer_vector?.prefill_trigger_positions || '-1');
            vConfig.prefill_trigger_tokens = parseListInput(config.steer_vector?.prefill_trigger_tokens || '');
            vConfig.generate_trigger_tokens = parseListInput(config.steer_vector?.generate_trigger_tokens || '');
            vConfig.normalize = config.steer_vector?.normalize || false;
            
            // 切换到新创建的标签页
            switchVectorTab(configId);
        }
        
        showMultiResponse({
            message: window.t('import_success_message'),
            description: window.t('import_success_description', { configName: configSelect.options[configSelect.selectedIndex].text })
        });
        
        // 清除选择框
        configSelect.value = '';
        
    } catch (error) {
        console.error('导入配置错误:', error);
        showMultiError(window.t('import_fail_error') + ': ' + error.message);
    }
} 