import React, { useState, useRef, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Alert,
  Paper,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material'
import {
  FlightTakeoff as DroneIcon,
  PlayArrow as PlayIcon,
  Warning as WarningIcon
} from '@mui/icons-material'
import { motion } from 'framer-motion'
import rockimg1 from "/rockimg1.jpeg";
import rockimg12 from "/rockimg12.jpeg";
import rockimg13 from "/rockimg13.jpeg";
import rockimg14 from "/rockimg14.jpeg";
// import { apiRequest, getApiUrl } from '../config/api' // Commented for frontend-only showcase

const Detection = () => {
  const [selectedImageIndex, setSelectedImageIndex] = useState(0)
  const [detectionResults, setDetectionResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showDemo, setShowDemo] = useState(false)
  const [showWarningDialog, setShowWarningDialog] = useState(false)
  const [imageDisplayDimensions, setImageDisplayDimensions] = useState({ 
    width: 0, 
    height: 0, 
    naturalWidth: 0, 
    naturalHeight: 0 
  })
  
  const imageRef = useRef(null)

  // Four static images with mock detection data
  const rockImages = [
    {
      src: rockimg1,
      name: 'Rock Image 1',
      detections: [
        { confidence: 0.89,bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0  },
        { confidence: 0.76, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0  },
        { confidence: 0.82, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0 }
      ]
    },
    {
      src: rockimg12,
      name: 'Rock Image 2',
      detections: [
        { confidence: 0.85, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0  },
      ]
    },
    {
      src: rockimg13,
      name: 'Rock Image 3',
      detections: [
        { confidence: 0.94, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0 },
        { confidence: 0.79, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0 },
        { confidence: 0.79, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0 },
        { confidence: 0.79, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0 }
      ]
    },
    {
      src: rockimg14,
      name: 'Rock Image 4',
      detections: [
        { confidence: 0.87, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0 },
        { confidence: 0.81, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0  },
        { confidence: 0.75, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0 },
        { confidence: 0.83, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0 },
        { confidence: 0.78, bbox: [0, 0, 0, 0], class: "rock", class_id: 0, area: 0 }
      ]
    }
  ]

  // Load demo detection on component mount
  useEffect(() => {
    loadStaticDetection()
  }, [selectedImageIndex])

  const loadStaticDetection = () => {
    console.log(`🔍 Loading static detection for image ${selectedImageIndex}...`)
    setLoading(true)
    setError(null)
    
    try {
      const currentImage = rockImages[selectedImageIndex]
      
      // Simulate processing delay for showcase
      setTimeout(() => {
        const mockDetectionResults = {
          detections: currentImage.detections,
          total_detections: currentImage.detections.length,
          confidence_threshold: 0.5,
          processing_time_ms: Math.random() * 300 + 150, // 150-450ms
          image_dimensions: { width: 640, height: 480 }, // Mock dimensions
          timestamp: new Date().toISOString()
        }
        
        console.log(`✅ Static detection loaded:`, mockDetectionResults)
        setDetectionResults(mockDetectionResults)
        setShowDemo(true)
        setLoading(false)
      }, 600) // Simulate processing time
      
    } catch (error) {
      console.error('❌ Static detection error:', error)
      setError('Failed to load detection results')
      setLoading(false)
    }
  }

  const tryDemoDetection = () => {
    // Show warning dialog first
    setShowWarningDialog(true)
  }

  const proceedWithDemo = () => {
    setShowWarningDialog(false)
    loadStaticDetection()
    setError(null)
  }

  const switchToNextImage = () => {
    const nextIndex = (selectedImageIndex + 1) % rockImages.length
    setSelectedImageIndex(nextIndex)
  }

  const handleImageLoad = () => {
    if (imageRef.current) {
      const { clientWidth, clientHeight, naturalWidth, naturalHeight } = imageRef.current
      setImageDisplayDimensions({ 
        width: clientWidth, 
        height: clientHeight,
        naturalWidth,
        naturalHeight
      })
      console.log('📐 Image loaded:', {
        displayed: { width: clientWidth, height: clientHeight },
        natural: { width: naturalWidth, height: naturalHeight }
      })
    }
  }

  const renderBoundingBoxes = () => {
    if (!detectionResults || !detectionResults.detections || !detectionResults.image_dimensions || 
        imageDisplayDimensions.width === 0 || imageDisplayDimensions.height === 0) {
      console.log('⚠️ Bounding boxes not rendered:', {
        hasDetectionResults: !!detectionResults,
        hasDetections: !!detectionResults?.detections,
        hasImageDimensions: !!detectionResults?.image_dimensions,
        displayDimensions: imageDisplayDimensions
      })
      return null
    }

    const { image_dimensions } = detectionResults
    
    // Calculate the actual displayed image size accounting for object-fit: contain
    const containerWidth = imageDisplayDimensions.width
    const containerHeight = imageDisplayDimensions.height
    const imageAspectRatio = image_dimensions.width / image_dimensions.height
    const containerAspectRatio = containerWidth / containerHeight
    
    let displayedImageWidth, displayedImageHeight, offsetX = 0, offsetY = 0
    
    if (imageAspectRatio > containerAspectRatio) {
      // Image is wider than container - fit to width
      displayedImageWidth = containerWidth
      displayedImageHeight = containerWidth / imageAspectRatio
      offsetY = (containerHeight - displayedImageHeight) / 2
    } else {
      // Image is taller than container - fit to height
      displayedImageHeight = containerHeight
      displayedImageWidth = containerHeight * imageAspectRatio
      offsetX = (containerWidth - displayedImageWidth) / 2
    }
    
    const scaleX = displayedImageWidth / image_dimensions.width
    const scaleY = displayedImageHeight / image_dimensions.height

    console.log(' Rendering bounding boxes:', {
      originalImageSize: image_dimensions,
      containerSize: { width: containerWidth, height: containerHeight },
      displayedImageSize: { width: displayedImageWidth, height: displayedImageHeight },
      offsets: { x: offsetX, y: offsetY },
      scaleFactors: { scaleX, scaleY },
      detectionsCount: detectionResults.detections.length
    })

    return (
      <svg
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none'
        }}
      >
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge> 
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        {detectionResults.detections.map((detection, index) => {
          const [x1, y1, x2, y2] = detection.bbox
          const scaledX = (x1 * scaleX) + offsetX
          const scaledY = (y1 * scaleY) + offsetY
          const scaledWidth = (x2 - x1) * scaleX
          const scaledHeight = (y2 - y1) * scaleY
          const confidence = (detection.confidence * 100).toFixed(1)
          const boxColor = confidence > 80 ? '#10b981' : confidence > 60 ? '#f59e0b' : '#ef4444'
          const textWidth = confidence.length * 8 + 35

          console.log(` Detection ${index + 1}:`, {
            originalBbox: [x1, y1, x2, y2],
            scaledBbox: [scaledX, scaledY, scaledX + scaledWidth, scaledY + scaledHeight],
            confidence: confidence + '%'
          })

          return (
            <g key={index}>
              {/* Bounding box rectangle with glow effect */}
              <rect
                x={scaledX}
                y={scaledY}
                width={scaledWidth}
                height={scaledHeight}
                fill="none"
                stroke={boxColor}
                strokeWidth="3"
                strokeDasharray="8,4"
                filter="url(#glow)"
                className="detection-box"
              />
              {/* Corner markers */}
              <circle cx={scaledX} cy={scaledY} r="4" fill={boxColor} />
              <circle cx={scaledX + scaledWidth} cy={scaledY} r="4" fill={boxColor} />
              <circle cx={scaledX} cy={scaledY + scaledHeight} r="4" fill={boxColor} />
              <circle cx={scaledX + scaledWidth} cy={scaledY + scaledHeight} r="4" fill={boxColor} />
              
              {/* Confidence label background */}
              <rect
                x={scaledX}
                y={scaledY - 28}
                width={textWidth}
                height="24"
                fill={boxColor}
                rx="12"
                style={{
                  opacity: 0.95
                }}
              />
              {/* Confidence label text */}
              <text
                x={scaledX + 6}
                y={scaledY - 10}
                fill="white"
                fontSize="11"
                fontWeight="bold"
                fontFamily="Arial, sans-serif"
              >
                🪨 {confidence}%
              </text>
            </g>
          )
        })}
      </svg>
    )
  }
  
  return (
    <Box>
      <style>
        {`
          @keyframes detectPulse {
            0% { stroke-opacity: 1; }
            50% { stroke-opacity: 0.6; }
            100% { stroke-opacity: 1; }
          }
          .detection-box {
            animation: detectPulse 2s infinite;
          }
        `}
      </style>
      
      {/* Drone-themed Header */}
      <motion.div
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Box sx={{ 
          mb: 4, 
          p: 3,
          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%)',
          borderRadius: 3,
          border: '1px solid rgba(59, 130, 246, 0.2)'
        }}>
          <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
            <motion.div
              animate={{ 
                rotate: [0, 5, -5, 0],
                scale: [1, 1.1, 1] 
              }}
              transition={{ 
                duration: 3, 
                repeat: Infinity,
                repeatType: "reverse" 
              }}
            >
              <DroneIcon sx={{ 
                fontSize: 48, 
                color: '#3b82f6'
              }} />
            </motion.div>
            <Box>
              <Typography variant="h3" component="h1" sx={{ 
                fontWeight: 800, 
                background: 'linear-gradient(45deg, #3b82f6, #10b981)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 0.5
              }}>
                🚁 Drone Rockfall Detection
              </Typography>
            </Box>
          </Stack>
          
          <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
             Advanced aerial surveillance system for real-time rockfall detection
          </Typography>
          
          <Button
            variant="contained"
            size="large"
            onClick={tryDemoDetection}
            startIcon={<PlayIcon />}
            sx={{
              background: 'linear-gradient(45deg, #3b82f6, #10b981)',
              color: 'white',
              fontWeight: 700,
              px: 4,
              py: 1.5
            }}
          >
            🚀 Try Live Demo Detection
          </Button>
        </Box>
      </motion.div>

      {/* Surveillance Status - Separate from images */}
      <Box sx={{ mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card sx={{ 
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(59, 130, 246, 0.3)',
              borderRadius: 3,
              height: 'fit-content'
            }}>
              <CardContent sx={{ p: 3 }}>
                <Paper
                  variant="outlined"
                  sx={{
                    p: 3,
                    textAlign: 'center',
                    borderStyle: 'dashed',
                    borderWidth: 2,
                    borderColor: showDemo ? '#10b981' : '#475569',
                    backgroundColor: showDemo ? 'rgba(16, 185, 129, 0.1)' : 'rgba(15, 23, 42, 0.5)',
                    borderRadius: 3
                  }}
                >
                  {showDemo ? (
                    <div>
                      <Typography variant="h4" sx={{ mb: 1, color: '#10b981', fontWeight: 700 }}>
                         Drone Surveillance Active
                      </Typography>
                    </div>
                  ) : (
                    <div>
                      <DroneIcon sx={{ fontSize: 48, color: '#64748b', mb: 2 }} />
                      <Typography variant="h4" sx={{ mb: 1, color: '#64748b', fontWeight: 700 }}>
                        Drone Surveillance
                      </Typography>
                    </div>
                  )}
                </Paper>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12}>
            <Card sx={{ 
              height: 'fit-content',
              background: 'rgba(15, 23, 42, 0.8)',
              border: detectionResults ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(59, 130, 246, 0.3)',
              borderRadius: 3
            }}>
              <CardContent>
                <Typography variant="h4" component="div" sx={{ fontWeight: 700, mb: 3, color: '#3b82f6' }}>
                  AI Detection Results
                </Typography>
                {detectionResults ? (
                  <Box>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Paper sx={{ 
                          p: 3, 
                          textAlign: 'center', 
                          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.2))',
                          border: '1px solid rgba(59, 130, 246, 0.3)',
                          borderRadius: 2
                        }}>
                          <Typography variant="h3" sx={{ 
                            fontWeight: 800, 
                            color: '#3b82f6'
                          }}>
                            {detectionResults.total_detections}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
                            🪨 Rocks Detected
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6}>
                        <Paper sx={{ 
                          p: 3, 
                          textAlign: 'center', 
                          background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.2))',
                          border: '1px solid rgba(16, 185, 129, 0.3)',
                          borderRadius: 2
                        }}>
                          <Typography variant="h3" sx={{ 
                            fontWeight: 800, 
                            color: '#10b981'
                          }}>
                            {detectionResults.detections && detectionResults.detections.length > 0 
                              ? (detectionResults.detections.reduce((sum, det) => sum + det.confidence, 0) / detectionResults.detections.length * 100).toFixed(1)
                              : 0}%
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
                            Detection Confidence
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                    {/* Detection Legend */}
                    {detectionResults.detections && detectionResults.detections.length > 0 && (
                      <Box sx={{ mt: 3 }}>
                        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                          Detection Legend
                        </Typography>
                        <Grid container spacing={1}>
                          <Grid item xs={4}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Box sx={{ 
                                width: 20, 
                                height: 20, 
                                borderRadius: 1, 
                                backgroundColor: '#10b981' 
                              }} />
                              <Typography variant="caption" color="text.secondary">
                                High (80%+)
                              </Typography>
                            </Box>
                          </Grid>
                          <Grid item xs={4}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Box sx={{ 
                                width: 20, 
                                height: 20, 
                                borderRadius: 1, 
                                backgroundColor: '#f59e0b' 
                              }} />
                              <Typography variant="caption" color="text.secondary">
                                Medium (60-80%)
                              </Typography>
                            </Box>
                          </Grid>
                          <Grid item xs={4}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Box sx={{ 
                                width: 20, 
                                height: 20, 
                                borderRadius: 1, 
                                backgroundColor: '#ef4444' 
                              }} />
                              <Typography variant="caption" color="text.secondary">
                                Low (&lt;60%)
                              </Typography>
                            </Box>
                          </Grid>
                        </Grid>
                      </Box>
                    )}
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4, opacity: 0.7 }}>
                    <DroneIcon sx={{ fontSize: 48, color: '#64748b', mb: 2 }} />
                    <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
                      🚁 Drone Ready for Analysis
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Click the demo button to see AI-powered rockfall detection
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      {/* Detailed Analysis Section */}
      {showDemo && (
        <Box sx={{ mb: 6 }}>
          <Typography variant="h4" sx={{ fontWeight: 700, mb: 3, color: '#3b82f6' }}>
            🔍 Detailed Analysis
          </Typography>
          <Card sx={{
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            borderRadius: 3,
            overflow: 'hidden'
          }}>
            <CardContent sx={{ p: 0 }}>
              <Box sx={{ 
                position: 'relative',
                backgroundColor: '#0f172a'
              }}>
                <img
                  ref={imageRef}
                  src={rockImages[selectedImageIndex].src}
                  alt={rockImages[selectedImageIndex].name}
                  onLoad={handleImageLoad}
                  style={{
                    width: '100%',
                    maxHeight: '600px',
                    objectFit: 'contain',
                    display: 'block'
                  }}
                />
                
                {/* Bounding boxes overlay */}
                {detectionResults && renderBoundingBoxes()}
                
                {/* Image title overlay */}
                <Box sx={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  background: 'linear-gradient(to top, rgba(0,0,0,0.8), transparent)',
                  p: 3
                }}>
                  <Typography variant="h5" sx={{ color: 'white', fontWeight: 600 }}>
                    {rockImages[selectedImageIndex].name}
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                    {detectionResults?.total_detections || 0} rocks detected • Avg. confidence: {detectionResults?.detections ? (detectionResults.detections.reduce((sum, det) => sum + det.confidence, 0) / detectionResults.detections.length * 100).toFixed(1) : 0}%
                  </Typography>
                </Box>

                {/* Live detection indicator */}
                {detectionResults && (
                  <Box sx={{
                    position: 'absolute',
                    top: 16,
                    left: 16,
                    backgroundColor: 'rgba(239, 68, 68, 0.9)',
                    color: 'white',
                    px: 3,
                    py: 1.5,
                    borderRadius: 2,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1
                  }}>
                    <Box sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      backgroundColor: '#fff',
                      animation: 'blink 1s infinite'
                    }} />
                    <Typography variant="body2" sx={{ fontWeight: 700 }}>
                      LIVE DETECTION
                    </Typography>
                  </Box>
                )}

                {/* Detection count badge */}
                {detectionResults && detectionResults.detections && detectionResults.detections.length > 0 && (
                  <Box sx={{
                    position: 'absolute',
                    top: 16,
                    right: 16,
                    backgroundColor: 'rgba(16, 185, 129, 0.9)',
                    color: 'white',
                    px: 3,
                    py: 1.5,
                    borderRadius: 2
                  }}>
                    <Typography variant="body2" sx={{ fontWeight: 700 }}>
                      🪨 {detectionResults.total_detections} DETECTED
                    </Typography>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Box>
      )}
      
      {/* All Detection Images - Grid Layout with click functionality */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 3, color: '#3b82f6' }}>
          🎯 All Detection Results
        </Typography>
        <Grid container spacing={3}>
          {rockImages.map((image, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card sx={{
                background: 'rgba(15, 23, 42, 0.8)',
                border: selectedImageIndex === index ? '3px solid #3b82f6' : '1px solid rgba(59, 130, 246, 0.3)',
                borderRadius: 3,
                overflow: 'hidden',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: '0 16px 32px rgba(59, 130, 246, 0.4)',
                  borderColor: '#3b82f6'
                }
              }}
              onClick={() => setSelectedImageIndex(index)}
              >
                <Box sx={{ position: 'relative' }}>
                  <img
                    src={image.src}
                    alt={image.name}
                    style={{
                      width: '100%',
                      height: '220px',
                      objectFit: 'cover',
                      display: 'block'
                    }}
                  />
                  
                  {/* Gradient overlay for better text readability */}
                  <Box sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '60px',
                    background: 'linear-gradient(to bottom, rgba(0,0,0,0.7), transparent)'
                  }} />
                  
                  {/* Detection count badge */}
                  <Box sx={{
                    position: 'absolute',
                    top: 12,
                    right: 12,
                    backgroundColor: 'rgba(16, 185, 129, 0.95)',
                    color: 'white',
                    px: 2,
                    py: 1,
                    borderRadius: 2,
                    backdropFilter: 'blur(10px)'
                  }}>
                    <Typography variant="body2" sx={{ fontWeight: 700, fontSize: '0.875rem' }}>
                      🪨 {image.detections.length}
                    </Typography>
                  </Box>

                  {/* Active indicator */}
                  {selectedImageIndex === index && (
                    <Box sx={{
                      position: 'absolute',
                      top: 12,
                      left: 12,
                      backgroundColor: 'rgba(59, 130, 246, 0.95)',
                      color: 'white',
                      px: 2,
                      py: 1,
                      borderRadius: 2,
                      backdropFilter: 'blur(10px)',
                      animation: 'pulse 2s infinite'
                    }}>
                      <Typography variant="body2" sx={{ fontWeight: 700, fontSize: '0.875rem' }}>
                        ✓ ACTIVE
                      </Typography>
                    </Box>
                  )}

                  {/* Confidence indicator */}
                  <Box sx={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    background: 'linear-gradient(to top, rgba(0,0,0,0.8), transparent)',
                    p: 2
                  }}>
                    <Typography variant="caption" sx={{ 
                      color: 'white', 
                      fontWeight: 600,
                      display: 'block'
                    }}>
                      Avg: {(image.detections.reduce((sum, det) => sum + det.confidence, 0) / image.detections.length * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </Box>
                
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" sx={{ 
                    fontWeight: 700, 
                    mb: 1,
                    color: selectedImageIndex === index ? '#3b82f6' : 'white'
                  }}>
                    {image.name}
                  </Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      {image.detections.length} rocks detected
                    </Typography>
                    <Box sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      backgroundColor: selectedImageIndex === index ? '#3b82f6' : '#10b981'
                    }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {/* Warning Dialog for Demo */}
      <Dialog
        open={showWarningDialog}
        onClose={() => setShowWarningDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 2,
          backgroundColor: '#fef3c7',
          color: '#d97706'
        }}>
          <WarningIcon />
          Drone Connection Warning
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <Typography variant="body1" sx={{ mb: 2 }}>
            ⚠️ <strong>No drones are currently configured or connected to the system.</strong>
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            This demo uses pre-recorded images for demonstration purposes. To use live drone surveillance:
          </Typography>
          <Box component="ul" sx={{ pl: 2, mb: 2 }}>
            <Typography component="li" variant="body2" color="text.secondary">
              Configure drone connection settings
            </Typography>
            <Typography component="li" variant="body2" color="text.secondary">
              Ensure drone is powered on and connected
            </Typography>
            <Typography component="li" variant="body2" color="text.secondary">
              Verify network connectivity
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary">
            Would you like to proceed with the demo using sample images?
          </Typography>
        </DialogContent>
        <DialogActions sx={{ p: 3 }}>
          <Button 
            onClick={() => setShowWarningDialog(false)}
            color="inherit"
          >
            Cancel
          </Button>
          <Button 
            onClick={proceedWithDemo}
            variant="contained"
            sx={{
              background: 'linear-gradient(45deg, #f59e0b, #d97706)',
              color: 'white'
            }}
          >
            Proceed with Demo
          </Button>
        </DialogActions>
      </Dialog>

      {/* Add custom styles for animations */}
      <style jsx>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0.3; }
        }
        @keyframes pulse {
          0% { transform: scale(1); }
          50% { transform: scale(1.05); }
          100% { transform: scale(1); }
        }
      `}</style>
    </Box>
  )
}

export default Detection
