import React from 'react';
import './QuickActions.css';

const QuickActions = ({ actionProvider }) => {
  const quickActions = [
    {
      id: 1,
      text: "� Camera System Status",
      action: () => actionProvider.getCameraStatus()
    },
    {
      id: 2,
      text: "🎯 Current Risk Assessment",
      action: () => actionProvider.getCurrentRiskStatus()
    },
    {
      id: 3,
      text: "� Recent Risk Alerts",
      action: () => actionProvider.getRecentAlerts()
    },
    {
      id: 4,
      text: "� System Overview",
      action: () => actionProvider.getSystemStatus()
    },
    {
      id: 5,
      text: "� Detection System Info",
      action: () => actionProvider.explainDetectionSystem()
    },
    {
      id: 6,
      text: "🗺️ DEM Analysis Guide",
      action: () => actionProvider.explainDEMAnalysis()
    },
    {
      id: 7,
      text: "🆘 Emergency Procedures",
      action: () => actionProvider.getEmergencyProcedures()
    }
  ];

  const handleQuickAction = (action, text) => {
    // Add user message to show what was clicked
    actionProvider.addMessage(
      actionProvider.createClientMessage(text)
    );
    
    // Execute the action
    setTimeout(() => {
      action();
    }, 300);
  };

  return (
    <div className="quick-actions-container">
      <div className="quick-actions-title">
        🏔️ Rockfall AI Assistant - Choose an option:
      </div>
      <div className="quick-actions-grid">
        {quickActions.map((item) => (
          <button
            key={item.id}
            className="quick-action-btn"
            onClick={() => handleQuickAction(item.action, item.text)}
          >
            {item.text}
          </button>
        ))}
      </div>
      <div className="quick-actions-footer">
        Or ask me about cameras, risk levels, alerts, detection system, or emergency procedures! 👇
      </div>
    </div>
  );
};

export default QuickActions;