// 聊天模块的JavaScript功能

// 存储预设信息
let chatPresets = {};
// 当前选中的预设
let currentPreset = "happy_mode";
// 存储聊天历史
let normalChatHistory = [];
let steeredChatHistory = [];

// 存储预设的备注信息
const presetNotes = {
    "happy_mode": "In scenarios requiring sadness, the model may become unusually cheerful.",
    "chinese": "The model will tend to respond in Chinese.",
    "reject_mode": "The model will tend to reject the request.",
    "cat_mode": "The model will mimic cat-like behavior."
};

/**
 * 初始化聊天界面
 */
export function initChat() {
    console.log("初始化Chat模块...");
    
    // 加载预设信息
    loadPresets();
    
    // 确保DOM加载完成后设置事件监听
    setTimeout(() => {
        // 设置事件监听
        setupEventListeners();
        
        // 设置默认预设为列表中的第一个
        const firstPresetItem = document.querySelector('.preset-item');
        if (firstPresetItem) {
            const firstPresetKey = firstPresetItem.getAttribute('onclick').match(/'([^']+)'/)[1];
            selectPreset(firstPresetKey);
        }
    }, 300);
}

/**
 * 加载预设列表
 */
async function loadPresets() {
    try {
        const response = await fetch(window.EasySteerConfig.getApiUrl('/api/chat/presets'));
        const data = await response.json();
        
        if (data.success && data.presets) {
            chatPresets = data.presets;
            console.log("预设加载成功:", chatPresets);
        } else {
            console.error("预设加载失败:", data.error || "未知错误");
            // 使用默认预设
            chatPresets = {
                "power_seeking": { name: "Power Seeking Mode", description: "A more assertive and powerful response style" },
                "chinese": { name: "Chinese Mode", description: "Responds in Chinese language" },
                "pirate": { name: "Pirate Mode", description: "Talks like a pirate" },
                "shakespeare": { name: "Shakespeare Mode", description: "Responds in Shakespearean English" },
                "poetry": { name: "Poetry Mode", description: "Responds in poetic verse" }
            };
        }
    } catch (error) {
        console.error("加载预设出错:", error);
        // 使用默认预设
        chatPresets = {
            "power_seeking": { name: "Power Seeking Mode", description: "A more assertive and powerful response style" },
            "chinese": { name: "Chinese Mode", description: "Responds in Chinese language" },
            "pirate": { name: "Pirate Mode", description: "Talks like a pirate" },
            "shakespeare": { name: "Shakespeare Mode", description: "Responds in Shakespearean English" },
            "poetry": { name: "Poetry Mode", description: "Responds in poetic verse" }
        };
    }
}

/**
 * 设置事件监听
 */
function setupEventListeners() {
    // 发送消息按钮
    const sendButton = document.getElementById('sendMessage');
    if (sendButton) {
        sendButton.addEventListener('click', sendChatMessage);
    }
    
    // 输入框按Enter发送消息
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
    
    // 重置聊天按钮
    const resetButton = document.getElementById('resetChat');
    if (resetButton) {
        resetButton.addEventListener('click', resetChat);
    }
    
    // 重启后端按钮事件在HTML中直接绑定了onclick="restartBackend()"
    
    // 预设点击事件
    const presetItems = document.querySelectorAll('.preset-item');
    presetItems.forEach(item => {
        item.addEventListener('click', function() {
            const presetName = this.dataset.preset;
            selectPreset(presetName);
        });
    });
    
    // Demo按钮仅作装饰，不添加事件监听
    
    // 重置设置按钮
    const resetSettingsBtn = document.getElementById('resetSettings');
    if (resetSettingsBtn) {
        resetSettingsBtn.addEventListener('click', resetSettings);
    }
}

/**
 * 选择预设
 */
export function selectPreset(preset) {
    // 设置当前预设
    currentPreset = preset;
    
    // 移除之前所有预设项的active类
    document.querySelectorAll('.preset-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // 为当前点击的预设项添加active类
    // 我们通过传递的preset key来找到正确的DOM元素
    const activeItem = document.querySelector(`.preset-item[data-preset='${preset}']`);
    if (activeItem) {
        activeItem.classList.add('active');
    }
    
    // 更新备注区域
    const notesContainer = document.getElementById('presetNotes');
    if (notesContainer) {
        notesContainer.textContent = presetNotes[preset] || "No notes for this mode.";
    }

    console.log(`已选择预设: ${preset}`);

    // 更新模型路径显示
    const modelPathInput = document.getElementById('modelPathChat');
    if (modelPathInput) {
        // Since model path is now handled by the backend, we can clear this or show preset name
        modelPathInput.value = preset;
    }
}

/**
 * 发送聊天消息
 */
export async function sendChatMessage() {
    try {
        const chatInput = document.getElementById('chatInput');
        if (!chatInput || !chatInput.value.trim()) return;
        
        const message = chatInput.value.trim();
        chatInput.value = '';
        
        // 添加用户消息到两个聊天窗口
        addMessageToChat('user', message, 'normalChatMessages');
        addMessageToChat('user', message, 'steeredChatMessages');
        
        // 在聊天窗口显示加载状态
        const normalLoadingId = addLoadingMessage('normalChatMessages');
        const steeredLoadingId = addLoadingMessage('steeredChatMessages');
        
        try {
            // 为当前对话创建历史记录（排除当前消息）
            const normalHistory = [...normalChatHistory];
            const steeredHistory = [...steeredChatHistory];
            
            const payload = {
                preset: currentPreset,
                message: message,
                history: normalHistory,  // 添加历史对话
                steered_history: steeredHistory, // 添加引导对话历史
                gpu_devices: document.getElementById('gpuDevicesChat').value || "0",
                temperature: parseFloat(document.getElementById('tempSetting').value) || 0,
                max_tokens: parseInt(document.getElementById('maxTokensChat').value) || 128,
                repetition_penalty: parseFloat(document.getElementById('repetitionSetting').value) || 1.1
            };
            
            // 调用API获取响应
            const response = await fetch(window.EasySteerConfig.getApiUrl('/api/chat'), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            
            if (data.success) {
                // 移除加载状态
                removeLoadingMessage(normalLoadingId);
                removeLoadingMessage(steeredLoadingId);
                
                // 添加AI响应
                addMessageToChat('ai', data.normal_response, 'normalChatMessages');
                addMessageToChat('ai', data.steered_response, 'steeredChatMessages');
                
                // 更新聊天历史
                normalChatHistory.push({role: 'user', content: message});
                normalChatHistory.push({role: 'assistant', content: data.normal_response});
                steeredChatHistory.push({role: 'user', content: message});
                steeredChatHistory.push({role: 'assistant', content: data.steered_response});
            } else {
                throw new Error(data.error || "获取响应失败");
            }
        } catch (error) {
            console.error("聊天请求出错:", error);
            
            // 移除加载状态
            removeLoadingMessage(normalLoadingId);
            removeLoadingMessage(steeredLoadingId);
            
            // 显示错误消息
            const errorMessage = `Error: ${error.message || "获取响应失败，请稍后再试"}`;
            addMessageToChat('ai', errorMessage, 'normalChatMessages');
            addMessageToChat('ai', errorMessage, 'steeredChatMessages');
        }
    } catch (e) {
        console.error("发送消息时发生错误:", e);
    }
}

/**
 * 添加消息到聊天窗口
 */
function addMessageToChat(role, content, containerId) {
    const messagesContainer = document.getElementById(containerId);
    if (!messagesContainer) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.classList.add(role === 'user' ? 'user-message' : 'ai-message');
    
    const messageBubble = document.createElement('div');
    messageBubble.classList.add('message-bubble');
    
    // 删除头像代码，不再显示"G"头像
    
    const contentSpan = document.createElement('span');
    contentSpan.textContent = content;
    messageBubble.appendChild(contentSpan);
    
    messageDiv.appendChild(messageBubble);
    messagesContainer.appendChild(messageDiv);
    
    // 滚动到底部
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * 添加加载消息
 */
function addLoadingMessage(containerId) {
    const messagesContainer = document.getElementById(containerId);
    if (!messagesContainer) return null;
    
    const messageId = `loading-${Date.now()}`;
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('ai-message');
    messageDiv.id = messageId;
    
    const messageBubble = document.createElement('div');
    messageBubble.classList.add('message-bubble');
    
    const loadingSpan = document.createElement('span');
    loadingSpan.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    messageBubble.appendChild(loadingSpan);
    
    messageDiv.appendChild(messageBubble);
    messagesContainer.appendChild(messageDiv);
    
    // 滚动到底部
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageId;
}

/**
 * 移除加载消息
 */
function removeLoadingMessage(messageId) {
    if (!messageId) return;
    const loadingMessage = document.getElementById(messageId);
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

/**
 * 重置聊天
 */
export function resetChat() {
    // 清空聊天历史
    normalChatHistory = [];
    steeredChatHistory = [];
    
    // 清空聊天窗口
    document.getElementById('normalChatMessages').innerHTML = '';
    document.getElementById('steeredChatMessages').innerHTML = '';
    
    console.log("聊天已重置");
}

/**
 * 重置设置
 */
export function resetSettings() {
    // GPU设备ID
    document.getElementById('gpuDevicesChat').value = '0';
    // 温度
    document.getElementById('tempSetting').value = '0';
    // 最大生成令牌数
    document.getElementById('maxTokensChat').value = '128';
    // 重复惩罚
    document.getElementById('repetitionSetting').value = '1.1';
    
    console.log("设置已重置");
} 