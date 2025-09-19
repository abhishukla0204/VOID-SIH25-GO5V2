import React, { useState, useEffect, useRef } from 'react'
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
  Close as CloseIcon
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
  
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  
  // Static connection status for showcase - no WebSocket needed
  const connectionStatus = 'Connected'
  const isConnected = true
  
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
      text: 'Rock Detection',
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
  
  // Render current page component
  const renderCurrentPage = () => {
    const pageProps = {
      systemStatus,
      connectionStatus,
      highRiskCount,
      onRiskDataUpdate: setCurrentRiskData
    }
    
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
  }
  
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
            sx={{ mr: 1 }}
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
                color: highRiskCount > 0 ? '#fbbf24' : 'inherit',
                filter: highRiskCount > 0 ? 'drop-shadow(0 0 8px rgba(251, 191, 36, 0.6))' : 'none'
              }} />
            </Badge>
          </IconButton>
          
          <Chip 
            label={connectionStatus}
            color={connectionStatus === 'Connected' ? 'success' : 'warning'}
            variant="outlined"
            size="small"
            sx={{ 
              borderColor: 'rgba(255,255,255,0.3)',
              color: 'white',
              fontSize: '0.75rem'
            }}
          />
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
        <Paper sx={{ width: 400, maxHeight: 500, overflow: 'auto' }}>
          <Box sx={{ p: 2, borderBottom: '1px solid #e0e0e0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">Risk Notifications ({riskAlerts.length})</Typography>
            {riskAlerts.length > 0 && (
              <Button size="small" onClick={clearAllNotifications} color="error">
                Clear All
              </Button>
            )}
          </Box>
          
          {riskAlerts.length === 0 ? (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                No notifications yet
              </Typography>
            </Box>
          ) : (
            <List sx={{ p: 0 }}>
              {riskAlerts.map((alert, index) => (
                <motion.div
                  key={alert.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <ListItem 
                    sx={{ 
                      borderBottom: index < riskAlerts.length - 1 ? '1px solid #f0f0f0' : 'none',
                      '&:hover': { 
                        backgroundColor: '#f5f5f5',
                        '& .MuiTypography-root': {
                          color: 'inherit !important'
                        },
                        '& .MuiChip-root': {
                          opacity: 1
                        }
                      }
                    }}
                  >
                    <Box sx={{ flex: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Chip 
                          label={alert.riskLevel} 
                          size="small" 
                          color={alert.riskLevel === 'HIGH' ? 'error' : 'warning'}
                        />
                        <Typography variant="caption" sx={{ color: '#888888' }}>
                          {alert.timestamp.toLocaleTimeString()}
                        </Typography>
                      </Box>
                      <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#1a1a1a' }}>
                        {alert.location}
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#666666' }}>
                        Risk: {(alert.riskScore * 100).toFixed(1)}% | {alert.type}
                      </Typography>
                    </Box>
                    <IconButton 
                      size="small" 
                      onClick={() => removeNotification(alert.id)}
                      sx={{ ml: 1 }}
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
    </Box>
  )
}

export default App