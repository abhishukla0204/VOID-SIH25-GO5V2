// Environment Configuration for Rockfall Detection System
// ======================================================

export const config = {
  // Application Settings
  app: {
    title: import.meta.env.VITE_APP_TITLE || 'Rockfall Detection System',
    version: import.meta.env.VITE_APP_VERSION || '1.0.0',
    environment: import.meta.env.VITE_ENVIRONMENT || 'development',
    mode: import.meta.env.VITE_MODE || 'standalone'
  },

  // Feature Flags
  features: {
    liveMonitoring: import.meta.env.VITE_ENABLE_LIVE_MONITORING === 'true',
    demAnalysis: import.meta.env.VITE_ENABLE_DEM_ANALYSIS === 'true',
    riskAssessment: import.meta.env.VITE_ENABLE_RISK_ASSESSMENT === 'true',
    detection: import.meta.env.VITE_ENABLE_DETECTION === 'true',
    debugMode: import.meta.env.VITE_ENABLE_DEBUG_MODE === 'true'
  },

  // UI Configuration
  ui: {
    defaultTheme: import.meta.env.VITE_DEFAULT_THEME || 'light',
    enableDarkMode: import.meta.env.VITE_ENABLE_DARK_MODE === 'true',
    autoRefreshInterval: parseInt(import.meta.env.VITE_AUTO_REFRESH_INTERVAL) || 5000
  },

  // Mock Data Settings
  data: {
    enableMockData: import.meta.env.VITE_ENABLE_MOCK_DATA === 'true',
    useStaticData: import.meta.env.VITE_USE_STATIC_DATA === 'true',
    simulateRealTime: import.meta.env.VITE_SIMULATE_REAL_TIME === 'true'
  },

  // Notification Settings
  notifications: {
    enabled: import.meta.env.VITE_NOTIFICATION_ENABLED === 'true',
    sound: import.meta.env.VITE_NOTIFICATION_SOUND === 'true',
    highRiskThreshold: parseInt(import.meta.env.VITE_HIGH_RISK_THRESHOLD) || 75,
    cooldown: parseInt(import.meta.env.VITE_NOTIFICATION_COOLDOWN) || 120000
  },

  // Camera Configuration
  camera: {
    enabled: import.meta.env.VITE_CAMERA_ENABLED === 'true',
    refreshRate: parseInt(import.meta.env.VITE_CAMERA_REFRESH_RATE) || 30,
    urls: {
      east: import.meta.env.VITE_CAMERA_EAST_URL || 'https://res.cloudinary.com/dyb6aumhm/video/upload/v1758167914/1_znxt5x.mp4',
      west: import.meta.env.VITE_CAMERA_WEST_URL || 'https://res.cloudinary.com/dyb6aumhm/video/upload/v1758167915/2_lrgtxq.mp4',
      north: import.meta.env.VITE_CAMERA_NORTH_URL || 'https://res.cloudinary.com/dyb6aumhm/video/upload/v1758167915/3_gk37sc.mp4'
    }
  },

  // Risk Assessment Settings
  risk: {
    updateInterval: parseInt(import.meta.env.VITE_RISK_UPDATE_INTERVAL) || 5000,
    forceHighChance: parseFloat(import.meta.env.VITE_RISK_FORCE_HIGH_CHANCE) || 0.2,
    calculationMode: import.meta.env.VITE_RISK_CALCULATION_MODE || 'enhanced'
  },

  // Detection Settings
  detection: {
    confidenceThreshold: parseFloat(import.meta.env.VITE_DETECTION_CONFIDENCE_THRESHOLD) || 0.7,
    demoMode: import.meta.env.VITE_DETECTION_DEMO_MODE === 'true'
  },

  // Performance Settings
  performance: {
    enableCache: import.meta.env.VITE_ENABLE_CACHE === 'true',
    debugPerformance: import.meta.env.VITE_DEBUG_PERFORMANCE === 'true'
  }
}

// Utility functions
export const isDevelopment = () => config.app.environment === 'development'
export const isProduction = () => config.app.environment === 'production'
export const isDebugMode = () => config.features.debugMode && isDevelopment()

// Feature check helpers
export const isFeatureEnabled = (featureName) => config.features[featureName] ?? false

// Log configuration in development
if (isDevelopment()) {
  console.log('🔧 Application Configuration:', config)
}

export default config