import React, { useState, useEffect, useRef } from 'react'
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material'
import {
  TrendingUp as TrendingUpIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  PhotoCamera as CameraIcon,
  Assessment as AssessmentIcon,
  WaterDrop as WaterDropIcon,
  Thermostat as ThermostatIcon,
  Terrain as TerrainIcon,
  Timeline as VibrationIcon,
  Shield as ShieldIcon
} from '@mui/icons-material'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { motion } from 'framer-motion'

const Dashboard = ({ systemStatus, connectionStatus, highRiskCount, environmentalData, onRiskDataUpdate }) => {
  const [recentActivities, setRecentActivities] = useState([])
  const [riskTrends, setRiskTrends] = useState([])
  const [detectionStats, setDetectionStats] = useState({
    totalDetections: 0,
    averageConfidence: 0,
    processedImages: 0
  })
  
  useEffect(() => {
    // Initialize with some historical mock data
    const initialRiskData = Array.from({ length: 23 }, (_, i) => ({
      hour: `${i}:00`,
      risk: Math.random() * 0.6, // Start with lower risk values
      detections: Math.floor(Math.random() * 3)
    }))
    
    // Add current hour as the last data point
    const currentHour = new Date().getHours()
    initialRiskData.push({
      hour: `${currentHour}:00`,
      risk: 0, // Will be updated by real data
      detections: 0
    })
    
    setRiskTrends(initialRiskData)
    
    const mockActivities = [
      { time: '14:30', type: 'detection', message: 'Rock detected with 95% confidence', severity: 'info' },
      { time: '14:15', type: 'risk', message: 'Risk level increased to MEDIUM', severity: 'warning' },
      { time: '13:45', type: 'system', message: 'All models loaded successfully', severity: 'success' },
      { time: '13:30', type: 'detection', message: '3 rocks detected in sector A', severity: 'info' },
    ]
    setRecentActivities(mockActivities)
    
    setDetectionStats({
      totalDetections: highRiskCount || 0,
      averageConfidence: 0.87,
      processedImages: Math.max(50, (highRiskCount || 0) * 12)
    })
  }, [highRiskCount])
  
  // Update graph when environmental data changes
  useEffect(() => {
    if (environmentalData && environmentalData.currentRisk !== undefined) {
      const currentHour = new Date().getHours()
      const currentRiskForGraph = environmentalData.currentRisk / 100 // Convert percentage to 0-1 scale for graph
      
      setRiskTrends(prevTrends => {
        const newTrends = [...prevTrends]
        // Update the current hour's data point
        const currentHourIndex = newTrends.findIndex(item => item.hour === `${currentHour}:00`)
        if (currentHourIndex !== -1) {
          newTrends[currentHourIndex] = {
            ...newTrends[currentHourIndex],
            risk: currentRiskForGraph,
            detections: environmentalData.riskLevel === 'HIGH' ? Math.floor(Math.random() * 3) + 2 : Math.floor(Math.random() * 2)
          }
        }
        return newTrends
      })
    }
  }, [environmentalData])
  
  const getStatusIcon = (status) => {
    switch (status) {
      case 'operational':
        return <CheckCircleIcon sx={{ color: '#10b981' }} />
      case 'warning':
        return <WarningIcon sx={{ color: '#f59e0b' }} />
      case 'error':
        return <ErrorIcon sx={{ color: '#ef4444' }} />
      default:
        return <WarningIcon sx={{ color: '#6b7280' }} />
    }
  }
  
  const getActivityIcon = (type) => {
    switch (type) {
      case 'detection':
        return <CameraIcon sx={{ color: '#8b5cf6' }} />
      case 'risk':
        return <AssessmentIcon sx={{ color: '#f59e0b' }} />
      case 'system':
        return <SecurityIcon sx={{ color: '#10b981' }} />
      default:
        return <CheckCircleIcon sx={{ color: '#6b7280' }} />
    }
  }
  
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'success': return '#10b981'
      case 'warning': return '#f59e0b'
      case 'error': return '#ef4444'
      default: return '#3b82f6'
    }
  }
  
  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 700, mb: 1 }}>
          System Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Real-time monitoring of rockfall detection and prediction system
        </Typography>
      </Box>
      
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SecurityIcon sx={{ color: '#3b82f6', mr: 1 }} />
                  <Typography variant="h6" component="div">
                    System Status
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {getStatusIcon('operational')}
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    Operational
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  All systems operational
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <MemoryIcon sx={{ color: '#10b981', mr: 1 }} />
                  <Typography variant="h6" component="div">
                    Models Loaded
                  </Typography>
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 600 }}>
                  5 / 5
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={100}
                  sx={{ mt: 1, height: 6, borderRadius: 3 }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TrendingUpIcon sx={{ color: '#8b5cf6', mr: 1 }} />
                  <Typography variant="h6" component="div">
                    Detections Today
                  </Typography>
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 600 }}>
                  {detectionStats.totalDetections}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {detectionStats.processedImages} images processed
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SpeedIcon sx={{ color: '#f59e0b', mr: 1 }} />
                  <Typography variant="h6" component="div">
                    Avg Confidence
                  </Typography>
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 600 }}>
                  {(detectionStats.averageConfidence * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Detection accuracy
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
      
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600, color: '#1976d2' }}>
        Environmental & Sensor Data
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6} lg={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card 
              sx={{ 
                background: 'linear-gradient(135deg, #2196F3, #1976D2)',
                color: 'white',
                height: '100%',
                transition: 'transform 0.3s',
                '&:hover': { transform: 'translateY(-4px)' }
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <WaterDropIcon sx={{ fontSize: 40, mr: 2 }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Rainfall
                  </Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {environmentalData.rainfall.toFixed(1)} mm
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Last 24 hours
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.45 }}
          >
            <Card 
              sx={{ 
                background: 'linear-gradient(135deg, #FF9800, #F57C00)',
                color: 'white',
                height: '100%',
                transition: 'transform 0.3s',
                '&:hover': { transform: 'translateY(-4px)' }
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <ThermostatIcon sx={{ fontSize: 40, mr: 2 }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Temperature
                  </Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {environmentalData.temperature.toFixed(1)}°C
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Current ambient
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <Card 
              sx={{ 
                background: 'linear-gradient(135deg, #795548, #5D4037)',
                color: 'white',
                height: '100%',
                transition: 'transform 0.3s',
                '&:hover': { transform: 'translateY(-4px)' }
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TerrainIcon sx={{ fontSize: 40, mr: 2 }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Fracture Density
                  </Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {environmentalData.fractureDensity.toFixed(2)}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Fractures/m²
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.55 }}
          >
            <Card 
              sx={{ 
                background: 'linear-gradient(135deg, #E91E63, #C2185B)',
                color: 'white',
                height: '100%',
                transition: 'transform 0.3s',
                '&:hover': { transform: 'translateY(-4px)' }
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <VibrationIcon sx={{ fontSize: 40, mr: 2 }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Seismic Activity
                  </Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {environmentalData.seismicActivity.toFixed(1)}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Magnitude (Richter)
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
      
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <Card 
              sx={{ 
                background: `linear-gradient(135deg, ${
                  environmentalData.currentRisk > 75 ? '#D32F2F, #B71C1C' :
                  environmentalData.currentRisk > 40 ? '#F57C00, #E65100' :
                  '#388E3C, #2E7D32'
                })`,
                color: 'white',
                textAlign: 'center',
                py: 3,
                transition: 'transform 0.3s',
                '&:hover': { transform: 'scale(1.02)' }
              }}
            >
              <CardContent>
                <Typography variant="h5" sx={{ fontWeight: 600, mb: 2 }}>
                  Current Risk Level
                </Typography>
                <Typography variant="h2" sx={{ fontWeight: 700, mb: 2 }}>
                  {environmentalData.currentRisk.toFixed(1)}%
                </Typography>
                <Typography variant="h6" sx={{ opacity: 0.9 }}>
                  {environmentalData.riskLevel} RISK
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
      
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} lg={8}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Typography variant="h6" component="div" sx={{ mb: 3 }}>
                  Risk Trends (24 Hours)
                </Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={riskTrends}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="hour" stroke="#94a3b8" />
                      <YAxis 
                        stroke="#94a3b8" 
                        domain={[0, 1]}
                        tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1e293b', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                        formatter={(value, name) => [
                          `${(value * 100).toFixed(1)}%`, 
                          name === 'risk' ? 'Risk Level' : 'Detections'
                        ]}
                        labelFormatter={(label) => `Time: ${label}`}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="risk" 
                        stroke="#3b82f6" 
                        fill="rgba(59, 130, 246, 0.2)"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <Card className="glass-card" sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" component="div" sx={{ mb: 2 }}>
                  System Status
                </Typography>
                
                {['YOLO Detector', 'Risk Analyzer', 'Seismic Monitor', 'Weather Predictor', 'Stability Assessor'].map((model) => (
                  <Box key={model} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <CheckCircleIcon sx={{ color: '#10b981', fontSize: 16, mr: 1 }} />
                    <Typography variant="body2">
                      {model}
                    </Typography>
                  </Box>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <Card className="glass-card">
          <CardContent>
            <Typography variant="h6" component="div" sx={{ mb: 2 }}>
              Recent Activities
            </Typography>
            
            <List>
              {recentActivities.map((activity, index) => (
                <React.Fragment key={index}>
                  <ListItem>
                    <ListItemIcon>
                      {getActivityIcon(activity.type)}
                    </ListItemIcon>
                    <ListItemText
                      primary={activity.message}
                      secondary={activity.time}
                      secondaryTypographyProps={{
                        color: getSeverityColor(activity.severity)
                      }}
                    />
                  </ListItem>
                  {index < recentActivities.length - 1 && (
                    <Divider variant="inset" component="li" sx={{ borderColor: '#334155' }} />
                  )}
                </React.Fragment>
              ))}
            </List>
          </CardContent>
        </Card>
      </motion.div>
    </Box>
  )
}

export default Dashboard