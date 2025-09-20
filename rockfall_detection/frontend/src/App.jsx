import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Container, 
  Box, 
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  IconButton,
  Badge,
  Chip,
  useMediaQuery,
  useTheme,
  Popover,
  Paper,
  Card,
  CardContent,
  Button
} from '@mui/material'
import {
  Dashboard as DashboardIcon,
  PhotoCamera as CameraIcon,
  Assessment as AssessmentIcon,
  Settings as SettingsIcon,
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Videocam as VideocamIcon,
  Terrain as TerrainIcon,
  Close as CloseIcon,
  AccountCircle as AccountCircleIcon
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'

// Import page components
import Dashboard from './pages/Dashboard'
import Detection from './pages/Detection'
import RiskAssessment from './pages/RiskAssessment'
import Settings from './pages/Settings'
import LiveMonitoring from './pages/LiveMonitoring'
import DEMAnalysis from './pages/DEMAnalysis'

// Import chatbot component
import RockfallChatbot from '../chatbot/RockfallChatbot'

const drawerWidth = 280

function App() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [systemStatus, setSystemStatus] = useState({
    status: 'loading',
    models_loaded: {},
    active_connections: 0
  })
  const [highRiskCount, setHighRiskCount] = useState(0)
  const [riskAlerts, setRiskAlerts] = useState([])
  const [notificationAnchor, setNotificationAnchor] = useState(null)
  const [currentRiskData, setCurrentRiskData] = useState({
    currentRisk: 0,
    riskLevel: 'LOW',
    riskScore: 0,
    shouldTriggerNotification: false
  })
  const [environmentalData, setEnvironmentalData] = useState({
    rainfall: 0,
    temperature: 0,
    fractureDensity: 0,
    seismicActivity: 0,
    currentRisk: 0,
    riskLevel: 'LOW',
    riskScore: 0
  })
  
  // Chatbot ref for proactive alerts
  const chatbotRef = useRef(null)
  
  // Mock camera feeds data for chatbot (in real app, this would come from LiveMonitoring)
  const [cameraFeeds] = useState({
    east: { name: 'East Camera', online: true, status: 'active', resolution: '1920x1080', fps: 30, detections: 2, recording: false },
    west: { name: 'West Camera', online: true, status: 'active', resolution: '1920x1080', fps: 30, detections: 1, recording: false },
    north: { name: 'North Camera', online: true, status: 'active', resolution: '1920x1080', fps: 30, detections: 3, recording: true },
    south: { name: 'South Camera', online: false, status: 'offline', resolution: '1920x1080', fps: 0, detections: 0, recording: false }
  })
  
  // Risk calculation refs
  const lastNotifiedRisk = useRef(false)
  const stableHighRiskData = useRef(null)
  const isInitialLoad = useRef(true)
  const lastEnvironmentalDataRef = useRef(null)

  // Function to check if environmental data has changed significantly
  const hasSignificantChange = useCallback((newData, oldData) => {
    if (!oldData) return true
    
    const threshold = 0.5 // Only update if values change by more than 0.5
    return (
      Math.abs(newData.rainfall - oldData.rainfall) > threshold ||
      Math.abs(newData.temperature - oldData.temperature) > threshold ||
      Math.abs(newData.fractureDensity - oldData.fractureDensity) > threshold ||
      Math.abs(newData.seismicActivity - oldData.seismicActivity) > threshold ||
      Math.abs(newData.currentRisk - oldData.currentRisk) > threshold ||
      newData.riskLevel !== oldData.riskLevel
    )
  }, [])
  
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  
  // Static connection status for showcase - no WebSocket needed
  const connectionStatus = 'Connected'
  const isConnected = true
  
  // Risk calculation logic - runs continuously regardless of current page
  useEffect(() => {
    const updateEnvironmentalData = () => {
      // Generate new random values every time
      const baseRainfall = 15 + Math.random() * 15
      const baseTemp = 20 + Math.random() * 8
      const baseFracture = 2.0 + Math.random() * 1.5
      const baseSeismic = Math.random() * 2
      
      // Calculate new risk score
      let riskScore = Math.min(1.0, (baseRainfall / 30 + baseFracture / 3.0 + baseSeismic / 3.5 + Math.random() * 0.1) / 3)
      
      // Force high risk scenario occasionally for demo
      if (Math.random() < (import.meta.env.VITE_RISK_FORCE_HIGH_CHANCE || 0.1)) {
        riskScore = 0.75 + Math.random() * 0.15
        console.log('🔥 Forcing high risk scenario:', (riskScore * 100).toFixed(1) + '%')
      }
      
      const riskPercentage = Math.round(riskScore * 100 * 10) / 10
      const newRiskLevel = riskPercentage > 75 ? 'HIGH' : riskPercentage > 40 ? 'MEDIUM' : 'LOW'

      let finalEnvironmentalData

      // Check if we currently have stable high risk data stored
      if (stableHighRiskData.current) {
        // If we have stable data, we need consecutive low readings to exit high risk mode
        if (riskPercentage <= 70) {  // Only exit when well below 75% to avoid oscillation
          // Risk dropped significantly - clear stable data and use new values
          stableHighRiskData.current = null
          finalEnvironmentalData = {
            rainfall: Math.round(baseRainfall * 10) / 10,
            temperature: Math.round(baseTemp * 10) / 10,
            fractureDensity: Math.round(baseFracture * 10) / 10,
            seismicActivity: Math.round(baseSeismic * 10) / 10,
            currentRisk: riskPercentage,
            riskLevel: newRiskLevel,
            riskScore: riskScore
          }
        } else {
          // Keep using stable high risk data regardless of new calculated values
          finalEnvironmentalData = stableHighRiskData.current
        }
      } else {
        // No stable data stored
        if (riskPercentage > 75) {
          // Entering high risk for first time - store and freeze values
          stableHighRiskData.current = {
            rainfall: Math.round(baseRainfall * 10) / 10,
            temperature: Math.round(baseTemp * 10) / 10,
            fractureDensity: Math.round(baseFracture * 10) / 10,
            seismicActivity: Math.round(baseSeismic * 10) / 10,
            currentRisk: riskPercentage,
            riskLevel: newRiskLevel,
            riskScore: riskScore
          }
          finalEnvironmentalData = stableHighRiskData.current
        } else {
          // Normal operation below 75% - use dynamic values
          finalEnvironmentalData = {
            rainfall: Math.round(baseRainfall * 10) / 10,
            temperature: Math.round(baseTemp * 10) / 10,
            fractureDensity: Math.round(baseFracture * 10) / 10,
            seismicActivity: Math.round(baseSeismic * 10) / 10,
            currentRisk: riskPercentage,
            riskLevel: newRiskLevel,
            riskScore: riskScore
          }
        }
      }

      // Only update state if there are significant changes to prevent unnecessary re-renders
      if (hasSignificantChange(finalEnvironmentalData, lastEnvironmentalDataRef.current)) {
        setEnvironmentalData(finalEnvironmentalData)
        lastEnvironmentalDataRef.current = finalEnvironmentalData
      }
      
      // Don't trigger notifications on initial load
      const shouldTrigger = !isInitialLoad.current && 
                          finalEnvironmentalData.currentRisk > 75 && 
                          !lastNotifiedRisk.current
      
      if (shouldTrigger) {
        lastNotifiedRisk.current = true // Mark that we've notified
      } else if (finalEnvironmentalData.currentRisk <= 70) { // Reset at 70% to match exit condition
        lastNotifiedRisk.current = false // Reset when risk drops significantly below 75%
      }
      
      // Mark that initial load is complete after first update
      if (isInitialLoad.current) {
        isInitialLoad.current = false
      }
      
      // Always update current risk data for notifications (this doesn't cause re-renders of page components)
      setCurrentRiskData({
        currentRisk: finalEnvironmentalData.currentRisk,
        riskLevel: finalEnvironmentalData.riskLevel,
        riskScore: finalEnvironmentalData.riskScore,
        shouldTriggerNotification: shouldTrigger
      })
    }
    
    updateEnvironmentalData()
    // Increased interval to reduce DOM rebuilding - only update every 10 seconds instead of 5
    const interval = setInterval(updateEnvironmentalData, import.meta.env.VITE_RISK_UPDATE_INTERVAL || 10000)
    
    return () => clearInterval(interval)
  }, []) // Empty dependency array - runs once and continues
  
  // Simulate high risk scenarios for showcase using current risk data
  useEffect(() => {
    const simulateRiskAlerts = () => {
      // Only trigger notifications when shouldTriggerNotification is true
      if (!currentRiskData.shouldTriggerNotification) {
        return
      }
      
      const scenarios = [
        { location: 'North Face', type: 'Slope Instability' },
        { location: 'East Ridge', type: 'Heavy Rainfall' },
        { location: 'Mining Zone A', type: 'Seismic Activity' },
        { location: 'South Wall', type: 'Freeze-Thaw Cycles' }
      ]
      
      const randomScenario = scenarios[Math.floor(Math.random() * scenarios.length)]
      
      // Use actual risk data from Dashboard
      const riskScore = currentRiskData.riskScore
      const riskLevel = currentRiskData.riskLevel
      const currentRisk = currentRiskData.currentRisk
      
      randomScenario.riskScore = riskScore
      randomScenario.riskLevel = riskLevel
      randomScenario.currentRisk = currentRisk
      randomScenario.timestamp = new Date()
      randomScenario.id = Date.now()
      
      // Add to alerts list (keep last 10)
      setHighRiskCount(prev => prev + 1)
      setRiskAlerts(prev => [randomScenario, ...prev.slice(0, 9)])
      
      // Show combined notification with dismiss option
      toast((t) => (
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', maxWidth: '400px' }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 'bold', marginBottom: '4px', color: '#dc2626' }}>
              🚨 {riskLevel} RISK ALERT
            </div>
            <div style={{ marginBottom: '8px', fontSize: '14px' }}>
              <strong>{randomScenario.location}</strong> - Risk: {currentRisk.toFixed(1)}%
              <br />Type: {randomScenario.type}
            </div>
            <div style={{ fontSize: '12px', color: '#6b7280', borderTop: '1px solid #e5e7eb', paddingTop: '6px' }}>
              📧 Email sent to safety officers<br />
              📱 SMS alerts dispatched to emergency team
            </div>
          </div>
          <button
            onClick={() => toast.dismiss(t.id)}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '18px',
              cursor: 'pointer',
              color: '#6b7280',
              padding: '0',
              lineHeight: '1'
            }}
          >
            ×
          </button>
        </div>
      ), {
        duration: 8000,
        style: {
          background: '#fef2f2',
          border: '1px solid #fecaca',
          color: '#374151'
        }
      })
      
      // Send proactive alert to chatbot
      if (chatbotRef.current && chatbotRef.current.sendProactiveAlert) {
        chatbotRef.current.sendProactiveAlert(randomScenario)
      }
    }
    
    // Only simulate alerts when risk data indicates high risk
    if (currentRiskData.shouldTriggerNotification) {
      // Trigger notification immediately
      simulateRiskAlerts()
    }
  }, [currentRiskData.shouldTriggerNotification, currentRiskData.currentRisk]) // Trigger when notification flag changes
  
  // Handle drawer toggle
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen)
  }
  
  // Handle notification panel
  const handleNotificationClick = (event) => {
    setNotificationAnchor(event.currentTarget)
  }
  
  const handleNotificationClose = () => {
    setNotificationAnchor(null)
  }
  
  const removeNotification = (notificationId) => {
    setRiskAlerts(prev => prev.filter(alert => alert.id !== notificationId))
  }
  
  const clearAllNotifications = () => {
    setRiskAlerts([])
    setHighRiskCount(0)
  }
  
  // Navigation items
  const navigationItems = [
    {
      text: 'Dashboard',
      icon: <DashboardIcon />,
      path: 'dashboard',
      color: '#3b82f6'
    },
    {
      text: 'Live Monitoring',
      icon: <VideocamIcon />,
      path: 'live-monitoring',
      color: '#ef4444'
    },
    {
      text: 'DEM Analysis',
      icon: <TerrainIcon />,
      path: 'dem-analysis',
      color: '#10b981'
    },
    {
      text: 'Rockfall Detection',
      icon: <CameraIcon />,
      path: 'detection',
      color: '#8b5cf6'
    },
    {
      text: 'Risk Assessment',
      icon: <AssessmentIcon />,
      path: 'risk-assessment',
      color: '#f59e0b'
    },
    {
      text: 'Settings',
      icon: <SettingsIcon />,
      path: 'settings',
      color: '#6b7280'
    }
  ]
  
  // Mock system status for showcase (instead of fetching from backend)
  useEffect(() => {
    // Set static operational status for showcase
    setSystemStatus({
      status: 'operational',
      models_loaded: {
        yolo_detector: true,
        risk_analyzer: true,
        seismic_monitor: true,
        weather_predictor: true,
        stability_assessor: true
      },
      active_connections: 5
    })
  }, [])
  
  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'operational': return 'success'
      case 'loading': return 'warning'
      case 'error': return 'error'
      default: return 'default'
    }
  }
  
  // Get connection status icon
  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'Connected':
        return <CheckCircleIcon sx={{ color: '#10b981', fontSize: 20 }} />
      case 'Connecting':
        return <WarningIcon sx={{ color: '#f59e0b', fontSize: 20 }} />
      default:
        return <WarningIcon sx={{ color: '#ef4444', fontSize: 20 }} />
    }
  }
  
  // Drawer content
  const drawerContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 3, borderBottom: '1px solid #334155' }}>
        <Typography variant="h5" component="div" sx={{ 
          fontWeight: 700,
          background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          mb: 1
        }}>
          🏔️ Rockfall AI
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Detection & Prediction System
        </Typography>
      </Box>
      
      {/* System Status */}
      <Box sx={{ p: 2, borderBottom: '1px solid #334155' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          {getConnectionIcon()}
          <Typography variant="body2" color="text.secondary">
            {connectionStatus}
          </Typography>
        </Box>
        <Chip 
          label={systemStatus.status}
          color={getStatusColor(systemStatus.status)}
          size="small"
          sx={{ fontSize: '0.75rem' }}
        />
      </Box>
      
      {/* Navigation */}
      <List sx={{ flex: 1, py: 1 }}>
        {navigationItems.map((item) => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              selected={currentPage === item.path}
              onClick={() => {
                setCurrentPage(item.path)
                if (isMobile) setMobileOpen(false)
              }}
              sx={{
                mx: 1,
                borderRadius: 2,
                mb: 0.5,
                '&.Mui-selected': {
                  backgroundColor: `${item.color}20`,
                  '&:hover': {
                    backgroundColor: `${item.color}30`,
                  }
                }
              }}
            >
              <ListItemIcon sx={{ 
                color: currentPage === item.path ? item.color : 'text.secondary',
                minWidth: 40 
              }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: '0.9rem',
                  fontWeight: currentPage === item.path ? 600 : 400,
                  color: currentPage === item.path ? item.color : 'text.primary'
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      
      {/* Footer */}
      <Box sx={{ p: 2, borderTop: '1px solid #334155' }}>
        <Typography variant="caption" color="text.secondary" display="block">
          v1.0.0 • Models Ready
        </Typography>
        <Typography variant="caption" color="text.secondary">
          5 / 5 models loaded
        </Typography>
      </Box>
    </Box>
  )
  
  // Memoized page props to prevent unnecessary re-renders
  const pageProps = useMemo(() => ({
    systemStatus,
    connectionStatus,
    highRiskCount,
    environmentalData,
    onRiskDataUpdate: setCurrentRiskData
  }), [systemStatus, connectionStatus, highRiskCount, environmentalData])
  
  // Render current page component
  const renderCurrentPage = useCallback(() => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard {...pageProps} />
      case 'live-monitoring':
        return <LiveMonitoring {...pageProps} />
      case 'dem-analysis':
        return <DEMAnalysis {...pageProps} />
      case 'detection':
        return <Detection {...pageProps} />
      case 'risk-assessment':
        return <RiskAssessment {...pageProps} />
      case 'settings':
        return <Settings {...pageProps} />
      default:
        return <Dashboard {...pageProps} />
    }
  }, [currentPage, pageProps])
  
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          zIndex: theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {navigationItems.find(item => item.path === currentPage)?.text || 'Dashboard'}
          </Typography>
          
          <IconButton 
            color="inherit" 
            sx={{ 
              mr: 1,
              backgroundColor: highRiskCount > 0 ? 'transparent' : '#10b981',
              '&:hover': {
                backgroundColor: highRiskCount > 0 ? 'rgba(255,255,255,0.1)' : '#059669'
              }
            }}
            onClick={handleNotificationClick}
          >
            <Badge 
              badgeContent={highRiskCount} 
              color="error"
              max={99}
              sx={{
                '& .MuiBadge-badge': {
                  backgroundColor: highRiskCount > 0 ? '#ef4444' : 'transparent',
                  animation: highRiskCount > 0 ? 'pulse 2s infinite' : 'none'
                }
              }}
            >
              <NotificationsIcon sx={{ 
                color: highRiskCount > 0 ? '#fbbf24' : 'white',
                filter: highRiskCount > 0 ? 'drop-shadow(0 0 8px rgba(251, 191, 36, 0.6))' : 'none'
              }} />
            </Badge>
          </IconButton>
          
          <IconButton 
            color="inherit"
            sx={{ 
              ml: 1,
              '&:hover': {
                backgroundColor: 'rgba(255,255,255,0.1)'
              }
            }}
          >
            <AccountCircleIcon sx={{ 
              fontSize: 32,
              color: 'white'
            }} />
          </IconButton>
        
        </Toolbar>
      </AppBar>

      {/* Notifications Panel */}
      <Popover
        open={Boolean(notificationAnchor)}
        anchorEl={notificationAnchor}
        onClose={handleNotificationClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <Paper sx={{ 
          width: 400, 
          maxHeight: 500, 
          overflow: 'auto',
          backgroundColor: '#1e293b',
          color: '#f8fafc',
          border: '1px solid #334155',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2)'
        }}>
          <Box sx={{ 
            p: 2, 
            borderBottom: '1px solid #475569', 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)'
          }}>
            <Typography variant="h6" sx={{ color: '#f1f5f9', fontWeight: 600 }}>
              Risk Notifications ({riskAlerts.length})
            </Typography>
            {riskAlerts.length > 0 && (
              <Button 
                size="small" 
                onClick={clearAllNotifications} 
                sx={{
                  color: '#ef4444',
                  borderColor: '#ef4444',
                  '&:hover': {
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderColor: '#f87171'
                  }
                }}
                variant="outlined"
              >
                Clear All
              </Button>
            )}
          </Box>
          
          {riskAlerts.length === 0 ? (
            <Box sx={{ 
              p: 4, 
              textAlign: 'center',
              background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)'
            }}>
              <Typography variant="body2" sx={{ 
                color: '#94a3b8',
                fontSize: '0.9rem',
                fontStyle: 'italic'
              }}>
                🔔 No notifications yet
              </Typography>
              <Typography variant="caption" sx={{ 
                color: '#64748b',
                display: 'block',
                mt: 1
              }}>
                You'll be notified when risk levels exceed 75%
              </Typography>
            </Box>
          ) : (
            <List sx={{ p: 0, backgroundColor: '#1e293b' }}>
              {riskAlerts.map((alert, index) => (
                <motion.div
                  key={alert.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <ListItem 
                    sx={{ 
                      borderBottom: index < riskAlerts.length - 1 ? '1px solid rgba(71, 85, 105, 0.3)' : 'none',
                      backgroundColor: 'transparent',
                      transition: 'all 0.2s ease-in-out',
                      position: 'relative',
                      overflow: 'hidden',
                      '&:hover': { 
                        backgroundColor: 'rgba(59, 130, 246, 0.08)',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: '3px',
                          backgroundColor: '#3b82f6',
                          transform: 'scaleY(1)',
                        },
                        '& .MuiTypography-root': {
                          color: 'inherit !important'
                        },
                        '& .MuiChip-root': {
                          opacity: 1,
                          transform: 'scale(1.05)'
                        },
                        '& .notification-close-btn': {
                          opacity: 1,
                          transform: 'scale(1.1)'
                        }
                      },
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        bottom: 0,
                        width: '3px',
                        backgroundColor: '#3b82f6',
                        transform: 'scaleY(0)',
                        transition: 'transform 0.2s ease-in-out',
                      },
                      cursor: 'pointer'
                    }}
                  >
                    <Box sx={{ flex: 1, py: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                        <Chip 
                          label={alert.riskLevel} 
                          size="small" 
                          sx={{
                            backgroundColor: alert.riskLevel === 'HIGH' ? '#ef4444' : '#f59e0b',
                            color: 'white',
                            fontWeight: 600,
                            fontSize: '0.75rem',
                            transition: 'transform 0.2s ease-in-out',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                          }}
                        />
                        <Typography variant="caption" sx={{ 
                          color: '#94a3b8',
                          fontSize: '0.75rem',
                          fontWeight: 500
                        }}>
                          {alert.timestamp.toLocaleTimeString()}
                        </Typography>
                      </Box>
                      <Typography variant="subtitle2" sx={{ 
                        fontWeight: 700, 
                        color: '#f1f5f9',
                        fontSize: '0.95rem',
                        mb: 0.5
                      }}>
                        📍 {alert.location}
                      </Typography>
                      <Typography variant="body2" sx={{ 
                        color: '#cbd5e1',
                        fontSize: '0.85rem',
                        lineHeight: 1.4
                      }}>
                        🚨 Risk: <span style={{ 
                          color: alert.riskLevel === 'HIGH' ? '#f87171' : '#fbbf24',
                          fontWeight: 600 
                        }}>{alert.currentRisk.toFixed(1)}%</span> | {alert.type}
                      </Typography>
                    </Box>
                    <IconButton 
                      size="small" 
                      onClick={() => removeNotification(alert.id)}
                      className="notification-close-btn"
                      sx={{ 
                        ml: 1,
                        opacity: 0.7,
                        transition: 'all 0.2s ease-in-out',
                        color: '#94a3b8',
                        '&:hover': {
                          backgroundColor: 'rgba(239, 68, 68, 0.2)',
                          color: '#ef4444'
                        }
                      }}
                    >
                      <CloseIcon fontSize="small" />
                    </IconButton>
                  </ListItem>
                </motion.div>
              ))}
            </List>
          )}
        </Paper>
      </Popover>
      
      {/* Navigation Drawer */}
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
              backgroundColor: '#1e293b',
              borderRight: '1px solid #334155'
            },
          }}
        >
          {drawerContent}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
              backgroundColor: '#1e293b',
              borderRight: '1px solid #334155'
            },
          }}
          open
        >
          {drawerContent}
        </Drawer>
      </Box>
      
      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          backgroundColor: '#0f172a',
        }}
      >
        <Toolbar />
        <Container maxWidth="xl" sx={{ py: 3 }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={currentPage}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {renderCurrentPage()}
            </motion.div>
          </AnimatePresence>
        </Container>
      </Box>
      
      {/* Rockfall AI Chatbot */}
      <RockfallChatbot
        ref={chatbotRef}
        environmentalData={environmentalData}
        systemStatus={systemStatus}
        riskAlerts={riskAlerts}
        cameraFeeds={cameraFeeds}
      />
    </Box>
  )
}

export default App