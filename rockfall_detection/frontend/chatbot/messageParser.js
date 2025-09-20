class MessageParser {
  constructor(actionProvider) {
    this.actionProvider = actionProvider;
  }

  parse(message) {
    console.log("User message received:", message);
    
    const lower = message.toLowerCase();

    // Camera and monitoring queries
    if (lower.includes("camera") || lower.includes("video") || lower.includes("feed") || lower.includes("monitoring")) {
      this.actionProvider.getCameraStatus();
    } 
    // Risk assessment queries
    else if (lower.includes("risk") || lower.includes("danger") || lower.includes("safe") || lower.includes("assessment")) {
      this.actionProvider.getCurrentRiskStatus();
    } 
    // System status queries
    else if (lower.includes("system") || lower.includes("status") || lower.includes("overview") || lower.includes("models") || lower.includes("ai")) {
      this.actionProvider.getSystemStatus();
    } 
    // Alert and notification queries
    else if (lower.includes("alert") || lower.includes("notification") || lower.includes("warning") || lower.includes("recent")) {
      this.actionProvider.getRecentAlerts();
    } 
    // Detection system queries
    else if (lower.includes("detection") || lower.includes("yolo") || lower.includes("rock") || lower.includes("object")) {
      this.actionProvider.explainDetectionSystem();
    } 
    // DEM analysis queries
    else if (lower.includes("dem") || lower.includes("elevation") || lower.includes("map") || lower.includes("terrain") || lower.includes("topography")) {
      this.actionProvider.explainDEMAnalysis();
    } 
    // Emergency procedures queries
    else if (lower.includes("emergency") || lower.includes("evacuation") || lower.includes("procedure") || lower.includes("protocol") || lower.includes("help")) {
      this.actionProvider.getEmergencyProcedures();
    } 
    // Environmental data queries
    else if (lower.includes("environment") || lower.includes("rainfall") || lower.includes("temperature") || lower.includes("seismic") || lower.includes("weather")) {
      this.actionProvider.getCurrentRiskStatus();
    } 
    // Greeting queries
    else if (lower.includes("hello") || lower.includes("hi") || lower.includes("hey") || lower.includes("start")) {
      this.actionProvider.handleGreeting();
    } 
    // Current status queries
    else if (lower.includes("current") || lower.includes("now") || lower.includes("today") || lower.includes("latest")) {
      this.actionProvider.getCurrentRiskStatus();
    } 
    // Generic help queries
    else if (lower.includes("what can you do") || lower.includes("features") || lower.includes("capabilities")) {
      this.actionProvider.handleGreeting();
    } 
    // Unknown message
    else {
      this.actionProvider.handleUnknownMessage(message);
    }
  }
}

export default MessageParser;
