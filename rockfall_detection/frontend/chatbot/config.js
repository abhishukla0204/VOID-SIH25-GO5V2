import { createChatBotMessage } from "react-chatbot-kit";
import React from "react";
import QuickActions from "./QuickActions";

const config = {
  botName: "🏔️ Rockfall AI Assistant",
  initialMessages: [
    createChatBotMessage("Hello! 👋 I'm your Rockfall AI Detection Assistant.", {
      widget: "quickActions"
    }),
    createChatBotMessage("I can help you with camera monitoring, risk assessment, detection results, and system status. Choose an option below or ask me anything about the rockfall detection system! 🚀"),
  ],
  state: {
    messages: [],
  },
  widgets: [
    {
      widgetName: "quickActions",
      widgetFunc: (props) => React.createElement(QuickActions, props),
    },
  ],
  // Custom settings to control behavior
  customStyles: {
    botMessageBox: {
      backgroundColor: "#1e293b",
    },
    chatButton: {
      backgroundColor: "#3b82f6",
    },
  },
};

export default config;