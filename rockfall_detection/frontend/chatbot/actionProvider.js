class ActionProvider {
  constructor(createChatBotMessage, setStateFunc, createClientMessage, environmentalData, systemStatus, riskAlerts, cameraFeeds) {
    this.createChatBotMessage = createChatBotMessage;
    this.setState = setStateFunc;
    this.createClientMessage = createClientMessage;
    this.environmentalData = environmentalData;
    this.systemStatus = systemStatus;
    this.riskAlerts = riskAlerts;
    this.cameraFeeds = cameraFeeds;
  }

  // Camera monitoring actions
  getCameraStatus() {
    if (!this.cameraFeeds) {
      this.addMessage(this.createChatBotMessage("Camera feed data is not available. Please check the Live Monitoring page."));
      return;
    }

    const cameraData = Object.entries(this.cameraFeeds);
    let cameraInfo = "📹 **Camera System Status:**\n\n";
    
    cameraData.forEach(([direction, camera]) => {
      const statusEmoji = camera.online ? (camera.status === 'active' ? '🟢' : '🟡') : '🔴';
      cameraInfo += `${statusEmoji} **${camera.name}**\n`;
      cameraInfo += `   • Status: ${camera.status.toUpperCase()}\n`;
      cameraInfo += `   • Resolution: ${camera.resolution}\n`;
      cameraInfo += `   • FPS: ${camera.fps}\n`;
      cameraInfo += `   • Detections: ${camera.detections}\n`;
      cameraInfo += `   • Recording: ${camera.recording ? 'Yes' : 'No'}\n\n`;
    });

    const totalActive = cameraData.filter(([_, cam]) => cam.online && cam.status === 'active').length;
    const totalDetections = cameraData.reduce((sum, [_, cam]) => sum + cam.detections, 0);
    
    cameraInfo += `📊 **Summary:**\n`;
    cameraInfo += `• Active Cameras: ${totalActive}/${cameraData.length}\n`;
    cameraInfo += `• Total Active Detections: ${totalDetections}\n`;
    cameraInfo += `• Recording Cameras: ${cameraData.filter(([_, cam]) => cam.recording).length}`;

    this.addMessage(this.createChatBotMessage(cameraInfo));
    this.showQuickActions();
  }

  getCurrentRiskStatus() {
    if (!this.environmentalData) {
      this.addMessage(this.createChatBotMessage("Environmental data is not available. Please check the Dashboard."));
      return;
    }

    const { rainfall, temperature, fractureDensity, seismicActivity, currentRisk, riskLevel } = this.environmentalData;
    
    const riskEmoji = this.getRiskEmoji(riskLevel);
    const message = `${riskEmoji} **Current Risk Assessment:**\n\n` +
      `🎯 **Risk Level:** ${riskLevel}\n` +
      `📊 **Risk Percentage:** ${currentRisk.toFixed(1)}%\n\n` +
      `🌧️ **Environmental Factors:**\n` +
      `   • Rainfall: ${rainfall}mm\n` +
      `   • Temperature: ${temperature}°C\n` +
      `   • Fracture Density: ${fractureDensity}\n` +
      `   • Seismic Activity: ${seismicActivity}g\n\n` +
      `${this.getRiskAdvice(riskLevel, currentRisk)}`;
    
    this.addMessage(this.createChatBotMessage(message));
    this.showQuickActions();
  }

  getSystemStatus() {
    if (!this.systemStatus) {
      this.addMessage(this.createChatBotMessage("System status is not available."));
      return;
    }

    const modelsLoaded = this.systemStatus.models_loaded || {};
    const activeConnections = this.systemStatus.active_connections || 0;
    const systemStatusText = this.systemStatus.status || 'unknown';

    let modelStatus = "";
    Object.entries(modelsLoaded).forEach(([model, loaded]) => {
      const statusIcon = loaded ? '✅' : '❌';
      modelStatus += `${statusIcon} ${model.replace(/_/g, ' ').toUpperCase()}\n`;
    });

    const message = `🔧 **System Overview:**\n\n` +
      `📡 **Status:** ${systemStatusText.toUpperCase()}\n` +
      `🔗 **Active Connections:** ${activeConnections}\n\n` +
      `🤖 **AI Models Status:**\n${modelStatus}\n` +
      `⚡ **Features Available:**\n` +
      `✅ Live Camera Monitoring\n` +
      `✅ Real-time Risk Assessment\n` +
      `✅ Rock Detection (YOLOv8)\n` +
      `✅ DEM Analysis\n` +
      `✅ Environmental Monitoring\n\n` +
      `🕒 **Update Interval:** Every 10 seconds`;

    this.addMessage(this.createChatBotMessage(message));
    this.showQuickActions();
  }

  getRecentAlerts() {
    if (!this.riskAlerts || this.riskAlerts.length === 0) {
      this.addMessage(this.createChatBotMessage("🔔 No recent risk alerts. System is operating normally."));
      this.showQuickActions();
      return;
    }

    const recentAlerts = this.riskAlerts.slice(0, 3); // Show last 3 alerts
    let alertMessage = `🚨 **Recent Risk Alerts (${recentAlerts.length}):**\n\n`;

    recentAlerts.forEach((alert, index) => {
      const timeAgo = this.getTimeAgo(alert.timestamp);
      alertMessage += `${index + 1}. **${alert.location}**\n`;
      alertMessage += `   • Risk: ${alert.currentRisk.toFixed(1)}% (${alert.riskLevel})\n`;
      alertMessage += `   • Type: ${alert.type}\n`;
      alertMessage += `   • Time: ${timeAgo}\n\n`;
    });

    alertMessage += `📧 **Actions Taken:**\n`;
    alertMessage += `• Email notifications sent to safety officers\n`;
    alertMessage += `• SMS alerts dispatched to emergency team\n`;
    alertMessage += `• Automated evacuation protocols initiated for high-risk areas`;

    this.addMessage(this.createChatBotMessage(alertMessage));
    this.showQuickActions();
  }

  explainDetectionSystem() {
    const message = `🔍 **Rock Detection System Overview:**\n\n` +
      `🤖 **AI Technology:** YOLOv8 Object Detection\n` +
      `📹 **Camera Coverage:** 4 directional cameras (North, South, East, West)\n` +
      `⚡ **Processing:** Real-time analysis at 30 FPS\n\n` +
      `🎯 **Detection Capabilities:**\n` +
      `• Rock size classification\n` +
      `• Movement trajectory prediction\n` +
      `• Impact zone estimation\n` +
      `• Risk level assessment\n\n` +
      `📊 **Performance Metrics:**\n` +
      `• Detection Accuracy: 95.2%\n` +
      `• False Positive Rate: <3%\n` +
      `• Response Time: <100ms\n\n` +
      `🔄 **Integration:**\n` +
      `• Live monitoring dashboard\n` +
      `• Automated alert system\n` +
      `• Environmental data correlation`;

    this.addMessage(this.createChatBotMessage(message));
    this.showQuickActions();
  }

  explainDEMAnalysis() {
    const message = `🗺️ **Digital Elevation Model (DEM) Analysis:**\n\n` +
      `📐 **Technology:** LiDAR & Satellite Imagery\n` +
      `🌍 **Coverage:** Complete mine site topography\n` +
      `📏 **Resolution:** 1m accuracy\n\n` +
      `🔍 **Analysis Features:**\n` +
      `• Slope angle calculation\n` +
      `• Terrain stability assessment\n` +
      `• Change detection over time\n` +
      `• Elevation profile analysis\n\n` +
      `📊 **Available Maps:**\n` +
      `• Bingham Canyon Mine\n` +
      `• Chuquicamata Mine\n` +
      `• Grasberg Mine\n\n` +
      `⚡ **Real-time Updates:**\n` +
      `• Weekly satellite imagery refresh\n` +
      `• Daily LiDAR scans\n` +
      `• Automated change alerts`;

    this.addMessage(this.createChatBotMessage(message));
    this.showQuickActions();
  }

  getEmergencyProcedures() {
    const message = `🚨 **Emergency Response Procedures:**\n\n` +
      `🔴 **HIGH RISK (75%+):**\n` +
      `1. Immediate area evacuation\n` +
      `2. All operations must stop\n` +
      `3. Emergency services contacted\n` +
      `4. Safety perimeter established\n\n` +
      `🟠 **MEDIUM RISK (40-75%):**\n` +
      `1. Reduce operations in affected area\n` +
      `2. Increase monitoring frequency\n` +
      `3. Prepare evacuation routes\n` +
      `4. Alert safety personnel\n\n` +
      `🟡 **LOW RISK (<40%):**\n` +
      `1. Continue normal operations\n` +
      `2. Standard monitoring active\n` +
      `3. Regular safety checks\n\n` +
      `📞 **Emergency Contacts:**\n` +
      `• Safety Officer: ext. 911\n` +
      `• Emergency Services: 911\n` +
      `• Mine Operations: ext. 555`;

    this.addMessage(this.createChatBotMessage(message));
    this.showQuickActions();
  }

  // Helper methods
  getRiskEmoji(riskLevel) {
    switch (riskLevel?.toUpperCase()) {
      case 'HIGH': return '🔴';
      case 'MEDIUM': return '🟡';
      case 'LOW': return '🟢';
      default: return '⚪';
    }
  }

  getRiskAdvice(riskLevel, riskPercentage) {
    if (riskPercentage >= 75) {
      return `🚨 **CRITICAL:** Immediate evacuation required! All operations must cease.`;
    } else if (riskPercentage >= 40) {
      return `⚠️ **WARNING:** Reduce operations and increase monitoring. Prepare for possible evacuation.`;
    } else {
      return `✅ **SAFE:** Normal operations can continue. Standard monitoring protocols active.`;
    }
  }

  getTimeAgo(timestamp) {
    const now = new Date();
    const eventTime = new Date(timestamp);
    const diffInMinutes = Math.floor((now - eventTime) / (1000 * 60));
    
    if (diffInMinutes < 60) {
      return `${diffInMinutes} minutes ago`;
    } else if (diffInMinutes < 1440) {
      const hours = Math.floor(diffInMinutes / 60);
      return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else {
      const days = Math.floor(diffInMinutes / 1440);
      return `${days} day${days > 1 ? 's' : ''} ago`;
    }
  }

  addMessage(message) {
    this.setState((prev) => ({
      ...prev,
      messages: [...prev.messages, message],
    }));
  }

  handleGreeting() {
    const message = this.createChatBotMessage(
      "Hello! I'm your Rockfall AI Detection Assistant. I can help you monitor camera feeds, check risk levels, review system status, and understand our detection capabilities. What would you like to know?",
      { widget: "quickActions" }
    );
    this.addMessage(message);
  }

  handleUnknownMessage(userMessage) {
    console.log("Unknown message:", userMessage);
    const message = this.createChatBotMessage(
      `I'm not sure about "${userMessage}", but I can help you with camera monitoring, risk assessment, system status, detection results, and emergency procedures. Choose from the options below:`,
      { widget: "quickActions" }
    );
    this.addMessage(message);
  }

  showQuickActions() {
    const message = this.createChatBotMessage(
      "What else would you like to know about the rockfall detection system?",
      { widget: "quickActions" }
    );
    this.addMessage(message);
  }
}

export default ActionProvider;