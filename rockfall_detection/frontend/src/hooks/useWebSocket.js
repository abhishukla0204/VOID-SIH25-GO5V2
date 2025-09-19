import { useState, useEffect, useRef } from 'react'
import { getWsUrl, getApiUrl } from '../config/api'
import useServerSentEvents from './useServerSentEvents'

const useWebSocket = (endpoint) => {
  const [connectionStatus, setConnectionStatus] = useState('Connecting')
  const [lastMessage, setLastMessage] = useState(null)
  const [currentUrl, setCurrentUrl] = useState(null)
  
  // Check environment variable to determine if we should use SSE only
  const useSSEOnly = import.meta.env.VITE_USE_SSE_ONLY === 'true'
  const [usingFallback, setUsingFallback] = useState(useSSEOnly) // Use SSE based on env var
  
  const ws = useRef(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 3
  
  // SSE fallback hook - now primary connection method when enabled
  const sseEndpoint = endpoint.replace('/ws', '/api/events/stream')
  const sseHook = useServerSentEvents(sseEndpoint)
  
  useEffect(() => {
    if (useSSEOnly) {
      // Skip WebSocket entirely for deployment, go directly to SSE
      console.log('🔄 Using SSE-only connection (WebSocket disabled via VITE_USE_SSE_ONLY)')
      setUsingFallback(true)
      setConnectionStatus('Connected via SSE')
      setCurrentUrl(sseEndpoint)
      
      // No WebSocket connection setup needed
      return () => {
        // No cleanup needed for WebSocket since we're not using it
      }
    }
    
    // If WebSocket is enabled, keep the original logic (for development)
    // This part can be kept for local development if needed
    console.log('⚠️ WebSocket mode is deprecated for deployment. Consider using SSE only.')
    setUsingFallback(true) // Force SSE even if WebSocket is enabled
    
  }, [endpoint, useSSEOnly, sseEndpoint])

  // Manual reconnect function
  const reconnect = () => {
    if (usingFallback) {
      sseHook.reconnect()
    } else {
      if (ws.current) {
        ws.current.close()
      }
      reconnectAttempts.current = 0
      setConnectionStatus('Connecting')
    }
  }

  // Send message function
  const sendMessage = (message) => {
    if (usingFallback) {
      // Use SSE sendMessage method
      return sseHook.sendMessage(message)
    } else if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
      return true
    }
    console.warn('No active connection. Cannot send message:', message)
    return false
  }

  // Return appropriate values based on connection type
  if (usingFallback) {
    return {
      connectionStatus: sseHook.connectionStatus,
      lastMessage: sseHook.lastMessage,
      currentUrl: sseHook.currentUrl,
      reconnect,
      sendMessage,
      isConnected: sseHook.isConnected,
      usingFallback: true,
      connectionType: 'SSE'
    }
  }

  return { 
    connectionStatus, 
    lastMessage, 
    currentUrl,
    reconnect,
    sendMessage,
    isConnected: connectionStatus === 'Connected',
    usingFallback: false,
    connectionType: 'WebSocket'
  }
}

export default useWebSocket