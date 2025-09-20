import React, { useState, useEffect, useRef } from 'react'
// import { getApiUrl, apiRequest } from '../config/api' // Commented for frontend-only showcase
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  IconButton,
  Button,
  FormControlLabel,
  Switch,
  Alert,
  Chip,
  Divider,
  Stack,
  Badge
} from '@mui/material'
import {
  Videocam as VideocamIcon,
  VideocamOff as VideocamOffIcon,
  Fullscreen as FullscreenIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  ErrorOutline as ErrorIcon,
  FiberManualRecord as RecordIcon,
  CameraAlt as CameraIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  RotateLeft as RotateLeftIcon,
  RotateRight as RotateRightIcon,
  PowerSettingsNew as PowerIcon
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'

const LiveMonitoring = () => {
  // Camera feed states with Cloudinary video URLs from environment
  const [cameraFeeds, setCameraFeeds] = useState({
    east: {
      id: 'camera-east',
      name: 'East Camera',
      status: 'active',
      videoUrl: import.meta.env.VITE_CAMERA_EAST_URL || 'https://res.cloudinary.com/dyb6aumhm/video/upload/v1758167914/1_znxt5x.mp4',
      lastUpdate: new Date(),
      resolution: '1920x1080',
      fps: import.meta.env.VITE_CAMERA_REFRESH_RATE || 30,
      detections: 2,
      recording: false,
      online: true
    },
    west: {
      id: 'camera-west',
      name: 'West Camera',
      status: 'active',
      videoUrl: import.meta.env.VITE_CAMERA_WEST_URL || 'https://res.cloudinary.com/dyb6aumhm/video/upload/v1758167915/2_lrgtxq.mp4',
      lastUpdate: new Date(),
      resolution: '1920x1080',
      fps: import.meta.env.VITE_CAMERA_REFRESH_RATE || 30,
      detections: 1,
      recording: false,
      online: true
    },
    north: {
      id: 'camera-north',
      name: 'North Camera',
      status: 'active',
      videoUrl: import.meta.env.VITE_CAMERA_NORTH_URL || 'https://res.cloudinary.com/dyb6aumhm/video/upload/v1758167915/3_gk37sc.mp4',
      lastUpdate: new Date(),
      resolution: '1920x1080',
      fps: import.meta.env.VITE_CAMERA_REFRESH_RATE || 30,
      detections: 3,
      recording: true,
      online: true
    },
    south: {
      id: 'camera-south',
      name: 'South Camera',
      status: 'offline',
      videoUrl: 'https://res.cloudinary.com/dyb6aumhm/video/upload/v1758167914/1_znxt5x.mp4', // Backup video
      lastUpdate: new Date(Date.now() - 300000), // 5 minutes ago
      resolution: '1920x1080',
      fps: 0,
      detections: 0,
      recording: false,
      online: false
    }
  })

  const [selectedCamera, setSelectedCamera] = useState(null)
  const [autoRefresh, setAutoRefresh] = useState(false) // Disabled by default for better performance
  const [fullscreenCamera, setFullscreenCamera] = useState(null)
  const [systemStats, setSystemStats] = useState({
    totalCameras: 4,
    activeCameras: 3,
    totalDetections: 6, // Sum of all detections
    storageUsed: '45.2 GB',
    uptime: '12h 34m'
  })

  const videoRefs = useRef({})

  // Toggle camera online/offline status
  const toggleCameraStatus = (direction) => {
    setCameraFeeds(prev => ({
      ...prev,
      [direction]: {
        ...prev[direction],
        online: !prev[direction].online,
        status: prev[direction].online ? 'offline' : 'active',
        fps: prev[direction].online ? 0 : 30,
        lastUpdate: new Date()
      }
    }))
  }

  // Simulate camera feed updates (for demo purposes)
  useEffect(() => {
    if (!autoRefresh) return

    const updateCameraFeeds = () => {
      // Simulate random detection updates for active cameras
      setCameraFeeds(prev => {
        const updated = { ...prev }
        Object.keys(updated).forEach(direction => {
          if (updated[direction].online && updated[direction].status === 'active') {
            // Randomly update detections for demonstration (less frequently)
            if (Math.random() > 0.9) { // Reduced from 0.8 to 0.9 for less frequent updates
              updated[direction] = {
                ...updated[direction],
                detections: Math.max(0, updated[direction].detections + (Math.random() > 0.5 ? 1 : -1)),
                lastUpdate: new Date()
              }
            }
          }
        })
        
        // Update system stats within the same state update
        const activeCameras = Object.values(updated).filter(cam => cam.online).length
        const totalDetections = Object.values(updated).reduce((sum, cam) => sum + cam.detections, 0)
        
        setSystemStats(prev => ({
          ...prev,
          activeCameras,
          totalDetections
        }))
        
        return updated
      })
    }

    const interval = setInterval(updateCameraFeeds, 10000) // Increased from 5s to 10s

    return () => clearInterval(interval)
  }, [autoRefresh]) // Removed cameraFeeds dependency to prevent cascading updates

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return '#4CAF50'
      case 'offline': return '#9E9E9E'
      case 'maintenance': return '#FF9800'
      case 'error': return '#F44336'
      default: return '#9E9E9E'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active': return <CheckCircleIcon sx={{ color: '#4CAF50' }} />
      case 'offline': return <VideocamOffIcon sx={{ color: '#9E9E9E' }} />
      case 'maintenance': return <WarningIcon sx={{ color: '#FF9800' }} />
      case 'error': return <ErrorIcon sx={{ color: '#F44336' }} />
      default: return <VideocamOffIcon sx={{ color: '#9E9E9E' }} />
    }
  }

  const handleCameraControl = (direction, action) => {
    console.log(`Camera ${direction}: ${action} action performed`)
    // Simulate control actions for showcase
    if (action === 'start' || action === 'stop') {
      // Already handled by toggle function
    } else if (action === 'refresh') {
      setCameraFeeds(prev => ({
        ...prev,
        [direction]: {
          ...prev[direction],
          lastUpdate: new Date()
        }
      }))
    }
  }

  const toggleRecording = (direction) => {
    const currentlyRecording = cameraFeeds[direction].recording
    const action = currentlyRecording ? 'stop_record' : 'record'
    
    handleCameraControl(direction, action)
    
    setCameraFeeds(prev => ({
      ...prev,
      [direction]: {
        ...prev[direction],
        recording: !prev[direction].recording
      }
    }))
  }

  const refreshCamera = (direction) => {
    handleCameraControl(direction, 'refresh')
    
    setCameraFeeds(prev => ({
      ...prev,
      [direction]: {
        ...prev[direction],
        lastUpdate: new Date()
      }
    }))
  }

  const enterFullscreen = (direction) => {
    console.log(`Entering fullscreen for ${direction} camera`)
    console.log('Camera data:', cameraFeeds[direction])
    setFullscreenCamera(direction)
  }

  const exitFullscreen = () => {
    setFullscreenCamera(null)
  }

  const CameraFeedCard = ({ direction, camera }) => (
    <Card 
      sx={{ 
        height: '100%',
        backgroundColor: '#1e293b',
        border: selectedCamera === direction ? '2px solid #3b82f6' : '1px solid #334155',
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.3)'
        }
      }}
    >
      <CardContent sx={{ p: 2 }}>
        {/* Camera Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CameraIcon sx={{ color: getStatusColor(camera.status) }} />
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'white' }}>
              {camera.name}
            </Typography>
            {camera.recording && (
              <RecordIcon sx={{ color: '#f44336', fontSize: 16, animation: 'blink 1s infinite' }} />
            )}
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip 
              size="small" 
              label={camera.status.toUpperCase()} 
              sx={{ 
                bgcolor: getStatusColor(camera.status), 
                color: 'white',
                fontWeight: 600,
                fontSize: '0.75rem'
              }} 
            />
            {getStatusIcon(camera.status)}
          </Box>
        </Box>

        {/* Video Feed Area */}
        <Box 
          sx={{ 
            width: '100%',
            aspectRatio: '16/9',
            backgroundColor: '#0f172a',
            borderRadius: 2,
            mb: 2,
            position: 'relative',
            overflow: 'hidden',
            cursor: camera.status === 'active' ? 'pointer' : 'default'
          }}
        >
          {camera.online && camera.status === 'active' ? (
            <>
              {/* Real video feed */}
              <video
                src={camera.videoUrl}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: 'block'
                }}
                autoPlay
                loop
                muted
                playsInline
                preload="metadata"
                onError={(e) => {
                  console.error(`Failed to load video for ${direction}:`, e)
                }}
                onLoadStart={() => {
                  console.log(`Video loading started for ${direction}`)
                }}
              />
                
              {/* Detection overlays */}
              {camera.detections > 0 && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 20,
                    left: 20,
                    border: '2px solid #f44336',
                    borderRadius: 1,
                    width: 80,
                    height: 60,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)'
                  }}
                >
                  <Typography variant="caption" sx={{ color: '#f44336', fontWeight: 600 }}>
                    ROCK
                  </Typography>
                </Box>
              )}

              {/* Timestamp overlay */}
              <Box
                sx={{
                  position: 'absolute',
                  bottom: 10,
                  right: 10,
                  backgroundColor: 'rgba(0,0,0,0.7)',
                  padding: '4px 8px',
                  borderRadius: 1
                }}
              >
                <Typography variant="caption" sx={{ color: 'white' }}>
                  {camera.lastUpdate.toLocaleTimeString()}
                </Typography>
              </Box>

              {/* Recording indicator */}
              {camera.recording && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 10,
                    right: 10,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.5,
                    backgroundColor: 'rgba(244, 67, 54, 0.9)',
                    padding: '4px 8px',
                    borderRadius: 1
                  }}
                >
                  <RecordIcon sx={{ fontSize: 12, color: 'white' }} />
                  <Typography variant="caption" sx={{ color: 'white', fontWeight: 600 }}>
                    REC
                  </Typography>
                </Box>
              )}
            </>
          ) : (
            <Box
              sx={{
                width: '100%',
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                gap: 2
              }}
            >
              <VideocamOffIcon sx={{ fontSize: 48, color: '#64748b' }} />
              <Typography variant="body2" sx={{ color: '#64748b', textAlign: 'center' }}>
                {camera.status === 'maintenance' ? 'Camera under maintenance' : 'Camera offline'}
              </Typography>
            </Box>
          )}
        </Box>

        {/* Camera Info */}
        <Stack spacing={1} sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="caption" sx={{ color: '#94a3b8' }}>Resolution:</Typography>
            <Typography variant="caption" sx={{ color: 'white' }}>{camera.resolution}</Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="caption" sx={{ color: '#94a3b8' }}>FPS:</Typography>
            <Typography variant="caption" sx={{ color: 'white' }}>{camera.fps}</Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="caption" sx={{ color: '#94a3b8' }}>Detections:</Typography>
            {/* <Badge badgeContent={camera.detections} color="error"> */}
              <Typography variant="caption" sx={{ color: 'white' }}>
                {camera.detections}
              </Typography>
            {/* </Badge> */}
          </Box>
        </Stack>

        {/* Camera Controls */}
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <IconButton 
            size="small" 
            onClick={() => toggleCameraStatus(direction)}
            sx={{ 
              color: camera.online ? '#10b981' : '#ef4444',
              '&:hover': { 
                color: camera.online ? '#059669' : '#dc2626',
                backgroundColor: camera.online ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)'
              },
              border: `1px solid ${camera.online ? '#10b981' : '#ef4444'}`,
              borderRadius: 1
            }}
            title={camera.online ? 'Turn camera offline' : 'Turn camera online'}
          >
            <PowerIcon fontSize="small" />
          </IconButton>
          <IconButton 
            size="small" 
            onClick={() => refreshCamera(direction)}
            sx={{ color: '#64748b', '&:hover': { color: '#3b82f6' } }}
            disabled={!camera.online}
          >
            <RefreshIcon fontSize="small" />
          </IconButton>
          <IconButton 
            size="small" 
            onClick={() => toggleRecording(direction)}
            sx={{ color: camera.recording ? '#f44336' : '#64748b' }}
            disabled={!camera.online}
          >
            <RecordIcon fontSize="small" />
          </IconButton>
          <IconButton 
            size="small" 
            onClick={() => enterFullscreen(direction)}
            disabled={!camera.online || camera.status !== 'active'}
            sx={{ color: '#64748b', '&:hover': { color: '#3b82f6' } }}
          >
            <FullscreenIcon fontSize="small" />
          </IconButton>
          <IconButton 
            size="small" 
            sx={{ color: '#64748b', '&:hover': { color: '#3b82f6' } }}
            disabled={!camera.online}
          >
            <SettingsIcon fontSize="small" />
          </IconButton>
        </Box>
      </CardContent>
    </Card>
  )

  return (
    <Box sx={{ minHeight: '100vh', backgroundColor: '#0f172a', color: 'white', p: 3 }}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" sx={{ fontWeight: 700, mb: 2, color: '#3b82f6' }}>
            🎥 Live Camera Monitoring
          </Typography>
          <Typography variant="body1" sx={{ color: '#94a3b8', mb: 1 }}>
            Real-time monitoring from four directional cameras with AI-powered rock detection
          </Typography>
          <Typography variant="body2" sx={{ color: '#8b5cf6', mb: 3, fontWeight: 600 }}>
            ⚡ Powered by YOLOv8
          </Typography>
          
          {/* System Stats */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={2.4}>
              <Paper sx={{ p: 2, backgroundColor: '#1e293b', textAlign: 'center' }}>
                <Typography variant="h6" sx={{ color: '#3b82f6', fontWeight: 600 }}>
                  {systemStats.activeCameras}/{systemStats.totalCameras}
                </Typography>
                <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                  Active Cameras
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <Paper sx={{ p: 2, backgroundColor: '#1e293b', textAlign: 'center' }}>
                <Typography variant="h6" sx={{ color: '#f44336', fontWeight: 600 }}>
                  {systemStats.totalDetections}
                </Typography>
                <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                  Active Detections
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <Paper sx={{ p: 2, backgroundColor: '#1e293b', textAlign: 'center' }}>
                <Typography variant="h6" sx={{ color: '#22c55e', fontWeight: 600 }}>
                  {systemStats.storageUsed}
                </Typography>
                <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                  Storage Used
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <Paper sx={{ p: 2, backgroundColor: '#1e293b', textAlign: 'center' }}>
                <Typography variant="h6" sx={{ color: '#8b5cf6', fontWeight: 600 }}>
                  {systemStats.uptime}
                </Typography>
                <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                  System Uptime
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <Paper sx={{ p: 2, backgroundColor: '#1e293b', textAlign: 'center' }}>
                <FormControlLabel
                  control={
                    <Switch 
                      checked={autoRefresh} 
                      onChange={(e) => setAutoRefresh(e.target.checked)}
                      color="primary"
                    />
                  }
                  label={
                    <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                      Auto Refresh
                    </Typography>
                  }
                />
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </motion.div>

      {/* Camera Feeds Grid */}
      <Grid container spacing={3}>
        {Object.entries(cameraFeeds).map(([direction, camera], index) => (
          <Grid item xs={12} md={6} key={direction}>
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1, duration: 0.5 }}
            >
              <CameraFeedCard direction={direction} camera={camera} />
            </motion.div>
          </Grid>
        ))}
      </Grid>

      {/* Fullscreen Modal */}
      <AnimatePresence>
        {fullscreenCamera && cameraFeeds[fullscreenCamera]?.online && cameraFeeds[fullscreenCamera]?.status === 'active' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.95)',
              zIndex: 9999,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 20
            }}
            onClick={exitFullscreen}
          >
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.8 }}
              style={{ maxWidth: '90vw', maxHeight: '90vh' }}
              onClick={(e) => e.stopPropagation()}
            >
              <Card sx={{ backgroundColor: '#1e293b' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h5" sx={{ color: 'white' }}>
                      {cameraFeeds[fullscreenCamera].name} - Fullscreen View
                    </Typography>
                    <Button onClick={exitFullscreen} sx={{ color: 'white' }}>
                      Exit Fullscreen
                    </Button>
                  </Box>
                  <Box 
                    sx={{ 
                      width: '80vw',
                      height: '60vh',
                      backgroundColor: '#0f172a',
                      borderRadius: 2,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      position: 'relative',
                      overflow: 'hidden'
                    }}
                  >
                    <video
                      src={cameraFeeds[fullscreenCamera]?.videoUrl || ''}
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        display: 'block'
                      }}
                      autoPlay
                      loop
                      muted
                      playsInline
                      controls
                      onError={(e) => {
                        console.error(`Failed to load fullscreen video for ${fullscreenCamera}:`, e)
                        // Fallback to placeholder if video fails to load
                        e.target.style.display = 'none'
                        if (e.target.nextSibling) {
                          e.target.nextSibling.style.display = 'flex'
                        }
                      }}
                      onLoadedData={(e) => {
                        // Hide fallback when video loads successfully
                        e.target.style.display = 'block'
                        if (e.target.nextSibling) {
                          e.target.nextSibling.style.display = 'none'
                        }
                      }}
                    />
                    
                    {/* Fallback content */}
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        display: 'none',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexDirection: 'column',
                        gap: 2
                      }}
                    >
                      <Typography variant="h4" sx={{ color: 'white' }}>
                        🎥 {fullscreenCamera.toUpperCase()} CAMERA - FULLSCREEN
                      </Typography>
                      <Typography variant="body1" sx={{ color: '#64748b' }}>
                        Loading video feed...
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Custom CSS for blinking animation
      <style jsx>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0.3; }
        }
      `}</style> */}
    </Box>
  )
}

export default LiveMonitoring