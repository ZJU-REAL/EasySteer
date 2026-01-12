/**
 * EasySteer Frontend Configuration
 * 
 * Centralized configuration for API endpoints and other settings.
 * This prevents hardcoded URLs and makes port changes easier.
 */

// Get the backend port from environment or use default
// Can be overridden by setting window.EASYSTEER_BACKEND_PORT before loading this script
const BACKEND_PORT = window.EASYSTEER_BACKEND_PORT || 5000;

// Configuration object
const EasySteerConfig = {
    // API Base URL - automatically uses current hostname with configured backend port
    apiBaseUrl: `${window.location.protocol}//${window.location.hostname}:${BACKEND_PORT}`,
    
    // API Endpoints (for reference)
    endpoints: {
        // Inference endpoints
        generate: '/api/generate',
        generateMulti: '/api/generate-multi',
        configs: '/api/configs',
        config: '/api/config',
        
        // Chat endpoints
        chat: '/api/chat',
        chatPresets: '/api/chat/presets',
        
        // Extraction endpoints
        extract: '/api/extract',
        extractStatus: '/api/extract-status',
        extractConfigs: '/api/extract-configs',
        extractConfig: '/api/extract-config',
        
        // Training endpoints
        train: '/api/train',
        trainStatus: '/api/train-status',
        trainConfigs: '/api/train-configs',
        trainConfig: '/api/train-config',
        
        // SAE endpoints
        saeSearch: '/api/sae/search',
        saeFeature: '/api/sae/feature',
        saeExtractVector: '/api/sae/extract-vector',
        
        // System endpoints
        restart: '/api/restart',
        health: '/api/health'
    },
    
    /**
     * Build a full API URL
     * @param {string} endpoint - The endpoint path (e.g., '/api/generate')
     * @returns {string} Full URL
     */
    getApiUrl: function(endpoint) {
        return this.apiBaseUrl + endpoint;
    },
    
    /**
     * Check if we're in development mode
     * @returns {boolean}
     */
    isDevelopment: function() {
        return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    },
    
    /**
     * Get configuration info
     * @returns {object} Configuration details
     */
    getInfo: function() {
        return {
            backendPort: BACKEND_PORT,
            apiBaseUrl: this.apiBaseUrl,
            hostname: window.location.hostname,
            protocol: window.location.protocol,
            isDevelopment: this.isDevelopment()
        };
    }
};

// Make it globally accessible
window.EasySteerConfig = EasySteerConfig;

// Log configuration in development mode
if (EasySteerConfig.isDevelopment()) {
    console.log('EasySteer Config:', EasySteerConfig.getInfo());
}

// Export for ES6 modules (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EasySteerConfig;
}
