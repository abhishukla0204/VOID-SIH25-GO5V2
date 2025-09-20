import React, { useState, useRef, forwardRef, useImperativeHandle, useEffect } from 'react';
import Chatbot from 'react-chatbot-kit';
import ActionProvider from './actionProvider';
import MessageParser from './messageParser';
import config from './config';
import './Chatbot.css';

const RockfallChatbot = forwardRef(({ environmentalData, systemStatus, riskAlerts, cameraFeeds }, ref) => {
  const [isOpen, setIsOpen] = useState(false);
  const chatbotRef = useRef();

  // Prevent auto-scroll when new messages are added
  useEffect(() => {
    if (isOpen) {
      // Prevent main page scroll when chatbot is open
      const handleWheel = (e) => {
        const chatWindow = document.querySelector('.chatbot-window');
        if (chatWindow && chatWindow.contains(e.target)) {
          e.stopPropagation();
        }
      };

      document.addEventListener('wheel', handleWheel, { passive: false });

      // Simple but effective approach - override scroll behavior completely
      const preventAutoScroll = () => {
        const messageContainer = document.querySelector('.react-chatbot-kit-chat-message-container');
        if (messageContainer) {
          let isUserScrolling = false;
          let savedScrollPosition = messageContainer.scrollTop;

          // Track when user is manually scrolling
          const handleScroll = () => {
            isUserScrolling = true;
            savedScrollPosition = messageContainer.scrollTop;
            
            // Clear the flag after user stops scrolling
            clearTimeout(window.userScrollTimeout);
            window.userScrollTimeout = setTimeout(() => {
              isUserScrolling = false;
            }, 150);
          };

          messageContainer.addEventListener('scroll', handleScroll);

          // Intercept any attempts to change scroll position
          const originalScrollTo = messageContainer.scrollTo.bind(messageContainer);
          const originalScrollTop = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollTop') || 
                                   Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'scrollTop');

          // Override scrollTo method
          messageContainer.scrollTo = function(x, y) {
            if (!isUserScrolling) {
              // Don't allow programmatic scrolling
              return;
            }
            originalScrollTo(x, y);
          };

          // Override scrollTop property
          if (originalScrollTop) {
            Object.defineProperty(messageContainer, 'scrollTop', {
              get: originalScrollTop.get,
              set: function(value) {
                if (!isUserScrolling) {
                  // Ignore programmatic scroll attempts
                  return;
                }
                originalScrollTop.set.call(this, value);
              },
              configurable: true
            });
          }

          return () => {
            messageContainer.removeEventListener('scroll', handleScroll);
            messageContainer.scrollTo = originalScrollTo;
            if (originalScrollTop) {
              Object.defineProperty(messageContainer, 'scrollTop', originalScrollTop);
            }
            clearTimeout(window.userScrollTimeout);
          };
        }
      };

      const timeout = setTimeout(preventAutoScroll, 100);

      return () => {
        document.removeEventListener('wheel', handleWheel);
        clearTimeout(timeout);
      };
    }
  }, [isOpen]);

  // Create wrapper classes that include the project data
  class WrappedActionProvider extends ActionProvider {
    constructor(createChatBotMessage, setStateFunc, createClientMessage) {
      super(createChatBotMessage, setStateFunc, createClientMessage, environmentalData, systemStatus, riskAlerts, cameraFeeds);
    }
  }

  class WrappedMessageParser extends MessageParser {
    constructor(actionProvider) {
      super(actionProvider);
    }
  }

  // Expose methods to parent component for proactive alerts
  useImperativeHandle(ref, () => ({
    sendProactiveAlert: (riskData) => {
      if (isOpen && chatbotRef.current) {
        const alertMessage = `🚨 **HIGH RISK ALERT** 🚨\n\n` +
          `📍 **Location:** ${riskData.location}\n` +
          `🎯 **Risk Level:** ${riskData.riskLevel}\n` +
          `📊 **Risk Percentage:** ${riskData.currentRisk.toFixed(1)}%\n` +
          `⚠️ **Type:** ${riskData.type}\n\n` +
          `🚨 **IMMEDIATE ACTION REQUIRED:**\n` +
          `• Evacuate affected area immediately\n` +
          `• Alert safety personnel\n` +
          `• Monitor all camera feeds\n` +
          `• Prepare emergency response team`;
        
        // Add the alert message to chatbot
        if (chatbotRef.current && chatbotRef.current.addMessage) {
          chatbotRef.current.addMessage({
            type: 'bot',
            message: alertMessage,
            timestamp: new Date()
          });
        }
      }
    },
    
    openChatbot: () => {
      setIsOpen(true);
    }
  }));

  return (
    <div className="rockfall-chatbot">
      {/* Chat Toggle Button */}
      <button 
        className={`chat-toggle-btn ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        title="🏔️ Rockfall AI Assistant - Click to chat!"
      >
        🤖
      </button>

      {/* Chatbot Window */}
      {isOpen && (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <h4>🏔️ Rockfall AI Assistant</h4>
            <button 
              className="close-btn"
              onClick={() => setIsOpen(false)}
              title="Close Assistant"
            >
              ✕
            </button>
          </div>
          
          <div className="chatbot-container">
            <div 
              className="custom-chatbot-wrapper"
              onScroll={(e) => e.stopPropagation()}
              style={{
                height: '100%',
                overflow: 'hidden'
              }}
            >
              <Chatbot
                ref={chatbotRef}
                config={config}
                actionProvider={WrappedActionProvider}
                messageParser={WrappedMessageParser}
                placeholderText="Type your message here..."
                headerText="🏔️ Rockfall AI Assistant"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

RockfallChatbot.displayName = 'RockfallChatbot';

export default RockfallChatbot;