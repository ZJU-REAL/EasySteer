// 通用工具函数模块

// Parse list input
export function parseListInput(input) {
    if (!input || input.trim() === '') {
        return null;
    }
    return input.split(',').map(item => parseInt(item.trim())).filter(num => !isNaN(num));
}

// Helper function to escape HTML
export function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Display status message
export function showStatus(message, type = 'info') {
    // Can use showResponse or showError directly
    if (type === 'error') {
        window.showError(message);
    } else {
        window.showResponse({ message: message });
    }
}

// 重启后端
export async function restartBackend() {
    const isConfirmed = confirm(window.t('confirm_restart'));
    if (!isConfirmed) {
        return;
    }
    
    try {
        showStatus(window.t('restarting_backend'), 'info');
        
        const response = await fetch(window.EasySteerConfig.getApiUrl('/api/restart'), {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            window.showResponse({
                message: window.t('restart_success_message'),
                description: window.t('restart_success_description')
            });
        } else {
            throw new Error(result.message || window.t('restart_fail_error'));
        }
        
    } catch (error) {
        window.showError(window.t('restart_fail_error') + ': ' + error.message);
    }
} 