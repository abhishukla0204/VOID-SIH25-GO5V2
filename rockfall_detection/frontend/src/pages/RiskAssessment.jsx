import React, { useState } from 'react'
// import { apiRequest } from '../config/api' // Commented for frontend-only showcase
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Alert,
  CircularProgress,
  Paper,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  LinearProgress
} from '@mui/material'
import {
  Assessment as AssessmentIcon,
  Terrain as TerrainIcon,
  WaterDrop as WaterIcon,
  Air as WindIcon,
  Thermostat as TempIcon,
  Speed as SpeedIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const RiskAssessment = () => {
  const [formData, setFormData] = useState({
    slope: 30,
    elevation: 1000,
    fracture_density: 2.5,
    roughness: 0.5,
    slope_variability: 0.3,
    instability_index: 0.4,
    wetness_index: 0.2,
    month: new Date().getMonth() + 1,
    day_of_year: Math.floor((new Date() - new Date(new Date().getFullYear(), 0, 0)) / 86400000),
    season: Math.floor((new Date().getMonth()) / 3),
    rainfall: 50,
    temperature: 20,
    temperature_variation: 5,
    freeze_thaw_cycles: 10,
    seismic_activity: 2,
    wind_speed: 10,
    precipitation_intensity: 5,
    humidity: 60,
    risk_score: 0.0
  })
  
  const [riskResults, setRiskResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [assessmentHistory, setAssessmentHistory] = useState([])
  
  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  // Enhanced Frontend risk calculation formula with proper weights
  const calculateRiskScore = (data) => {
    console.log('🧮 Calculating risk score with data:', data)
    
    // ===== PRIMARY RISK FACTORS (60% total weight) =====
    
    // 1. Slope Risk (25% - Most critical factor)
    let slopeRisk = 0
    if (data.slope <= 30) {
      slopeRisk = 0.1 // Very low risk for gentle slopes
    } else if (data.slope <= 45) {
      slopeRisk = 0.3 // Moderate risk
    } else if (data.slope <= 60) {
      slopeRisk = 0.6 // High risk - exponential increase
    } else if (data.slope <= 75) {
      slopeRisk = 0.85 // Very high risk
    } else {
      slopeRisk = 0.95 // Critical risk for near-vertical slopes
    }
    
    // 2. Geological Stability (20% weight)
    let geoStabilityRisk = 0
    // Fracture density is critical - more fractures = exponentially higher risk
    const fractureRisk = Math.min(Math.pow(data.fracture_density / 5, 1.5), 1) * 0.5
    const instabilityRisk = Math.pow(data.instability_index, 2) * 0.3 // Quadratic response
    const roughnessRisk = (1 - data.roughness) * 0.2 // Smooth surfaces are more dangerous
    geoStabilityRisk = fractureRisk + instabilityRisk + roughnessRisk
    
    // 3. Water/Moisture Impact (15% weight)
    let waterRisk = 0
    const rainfallRisk = Math.min(data.rainfall / 150, 1) * 0.4 // Heavy rain destabilizes
    const wetnessRisk = Math.pow(data.wetness_index, 1.5) * 0.35 // Exponential moisture effect
    const precipitationRisk = Math.min(data.precipitation_intensity / 15, 1) * 0.25
    waterRisk = rainfallRisk + wetnessRisk + precipitationRisk
    
    // ===== SECONDARY RISK FACTORS (25% total weight) =====
    
    // 4. Thermal Stress (12% weight)
    let thermalRisk = 0
    const tempVariationRisk = Math.min(data.temperature_variation / 20, 1) * 0.6
    const freezeThawRisk = Math.min(data.freeze_thaw_cycles / 30, 1) * 0.4
    thermalRisk = tempVariationRisk + freezeThawRisk
    
    // 5. Seismic Activity (8% weight)
    let seismicRisk = 0
    if (data.seismic_activity <= 2) {
      seismicRisk = 0.1
    } else if (data.seismic_activity <= 4) {
      seismicRisk = 0.4
    } else if (data.seismic_activity <= 6) {
      seismicRisk = 0.7
    } else {
      seismicRisk = 0.95 // Major earthquakes
    }
    
    // 6. Weather Conditions (5% weight)
    let weatherRisk = 0
    const windRisk = Math.min(data.wind_speed / 80, 1) * 0.5
    const humidityRisk = Math.min(data.humidity / 100, 1) * 0.3
    const elevationRisk = Math.min(data.elevation / 3000, 1) * 0.2 // Higher altitude = more exposure
    weatherRisk = windRisk + humidityRisk + elevationRisk
    
    // ===== SEASONAL MODIFIERS (15% total weight) =====
    
    // 7. Seasonal Risk (10% weight)
    let seasonalRisk = 0
    const month = data.month
    if (month >= 3 && month <= 5) {
      seasonalRisk = 0.8 // Spring - snowmelt and rain
    } else if (month >= 6 && month <= 8) {
      seasonalRisk = 0.3 // Summer - more stable
    } else if (month >= 9 && month <= 11) {
      seasonalRisk = 0.7 // Fall - freeze-thaw begins
    } else {
      seasonalRisk = 0.6 // Winter - freeze-thaw cycles
    }
    
    // 8. Temporal Factors (5% weight)
    let temporalRisk = 0
    // Day of year effect (spring/fall peaks)
    const dayRisk = Math.abs(Math.sin((data.day_of_year - 80) * Math.PI / 182.5)) * 0.6
    const variabilityRisk = data.slope_variability * 0.4 // Irregular slopes more dangerous
    temporalRisk = dayRisk + variabilityRisk
    
    // ===== FINAL WEIGHTED CALCULATION =====
    
    const primaryRisk = (slopeRisk * 0.25) + (geoStabilityRisk * 0.20) + (waterRisk * 0.15)
    const secondaryRisk = (thermalRisk * 0.12) + (seismicRisk * 0.08) + (weatherRisk * 0.05)
    const seasonalModifier = (seasonalRisk * 0.10) + (temporalRisk * 0.05)
    
    const baseScore = primaryRisk + secondaryRisk + seasonalModifier
    
    // ===== CRITICAL THRESHOLD ADJUSTMENTS =====
    
    let finalScore = baseScore
    
    // Critical slope adjustment - slopes >70° get major boost
    if (data.slope > 70) {
      finalScore = Math.min(finalScore + 0.3, 1)
    } else if (data.slope > 60) {
      finalScore = Math.min(finalScore + 0.15, 1)
    }
    
    // High fracture density boost
    if (data.fracture_density > 7) {
      finalScore = Math.min(finalScore + 0.2, 1)
    }
    
    // Major seismic activity override
    if (data.seismic_activity > 7) {
      finalScore = Math.min(finalScore + 0.25, 1)
    }
    
    // Extreme weather conditions
    if (data.rainfall > 200 || data.precipitation_intensity > 20) {
      finalScore = Math.min(finalScore + 0.15, 1)
    }
    
    console.log(`🔍 Risk breakdown: Slope=${slopeRisk.toFixed(2)}, Geo=${geoStabilityRisk.toFixed(2)}, Water=${waterRisk.toFixed(2)}, Final=${finalScore.toFixed(2)}`)
    
    return Math.min(Math.max(finalScore, 0), 1) // Clamp between 0 and 1
  }

  const getRiskLevel = (score) => {
    if (score >= 0.75) return 'CRITICAL'
    if (score >= 0.55) return 'HIGH'
    if (score >= 0.35) return 'MEDIUM'
    if (score >= 0.15) return 'LOW'
    return 'MINIMAL'
  }

  const generateRecommendations = (score, data) => {
    const recommendations = []
    
    // Base recommendations by risk level
    if (score >= 0.75) {
      recommendations.push('🚨 IMMEDIATE EVACUATION of all personnel from danger zones')
      recommendations.push('🛑 COMPLETE SHUTDOWN of operations in affected areas')
      recommendations.push('📡 Deploy emergency real-time monitoring systems')
      recommendations.push('🚁 Prepare emergency response and rescue teams')
    } else if (score >= 0.55) {
      recommendations.push('⚠️ Restrict access to high-risk slope areas')
      recommendations.push('📊 Increase monitoring frequency to every 15 minutes')
      recommendations.push('👥 Reduce personnel in vulnerable zones to essential only')
      recommendations.push('📞 Alert emergency response teams for standby')
    } else if (score >= 0.35) {
      recommendations.push('🔍 Conduct detailed geological slope stability analysis')
      recommendations.push('⏰ Increase inspection frequency to hourly checks')
      recommendations.push('📋 Review and update emergency response protocols')
      recommendations.push('🎯 Focus monitoring on identified high-risk zones')
    } else if (score >= 0.15) {
      recommendations.push('📅 Maintain standard monitoring schedule')
      recommendations.push('🛡️ Continue routine safety inspections')
      recommendations.push('📈 Monitor weather conditions for changes')
    } else {
      recommendations.push('✅ Continue normal operations with standard precautions')
      recommendations.push('📝 Maintain routine documentation and reporting')
    }
    
    // Specific factor-based recommendations
    if (data.slope > 70) {
      recommendations.push('🏔️ CRITICAL: Extreme slope angle detected - implement immediate stabilization')
    } else if (data.slope > 60) {
      recommendations.push('⛰️ HIGH SLOPE RISK: Consider terracing or retaining wall installation')
    } else if (data.slope > 45) {
      recommendations.push('📐 Moderate slope risk - monitor for slope movement indicators')
    }
    
    if (data.fracture_density > 7) {
      recommendations.push('🪨 CRITICAL FRACTURING: Rock mass highly unstable - urgent stabilization needed')
    } else if (data.fracture_density > 4) {
      recommendations.push('🔍 High fracture density - implement rock bolting or mesh installation')
    }
    
    if (data.rainfall > 200) {
      recommendations.push('🌧️ EXTREME RAINFALL: Install enhanced drainage and water diversion systems')
    } else if (data.rainfall > 100) {
      recommendations.push('💧 Heavy rainfall detected - improve slope drainage immediately')
    }
    
    if (data.seismic_activity > 7) {
      recommendations.push('🌍 MAJOR SEISMIC RISK: Earthquake activity detected - emergency protocols activated')
    } else if (data.seismic_activity > 4) {
      recommendations.push('📳 Elevated seismic activity - install vibration monitoring sensors')
    }
    
    if (data.freeze_thaw_cycles > 25) {
      recommendations.push('🧊 Extreme freeze-thaw cycles - monitor for thermal stress fractures')
    } else if (data.freeze_thaw_cycles > 15) {
      recommendations.push('❄️ Significant thermal cycling - inspect for weathering damage')
    }
    
    if (data.temperature_variation > 25) {
      recommendations.push('🌡️ Extreme temperature variations causing thermal stress')
    }
    
    if (data.precipitation_intensity > 20) {
      recommendations.push('⛈️ SEVERE WEATHER: Intense precipitation causing rapid saturation')
    }
    
    if (data.wind_speed > 80) {
      recommendations.push('💨 High wind speeds may affect loose debris and monitoring equipment')
    }
    
    return recommendations
  }
  
  const handleAssessment = () => {
    setLoading(true)
    setError(null)
    
    try {
      // Simulate processing delay for showcase
      setTimeout(() => {
        const riskScore = calculateRiskScore(formData)
        const riskLevel = getRiskLevel(riskScore)
        const recommendations = generateRecommendations(riskScore, formData)
        
        // Identify contributing risk factors based on enhanced analysis
        const contributingFactors = []
        if (formData.slope > 60) contributingFactors.push('steep_slope')
        if (formData.slope > 75) contributingFactors.push('critical_slope')
        if (formData.fracture_density > 5) contributingFactors.push('high_fracture_density')
        if (formData.fracture_density > 7) contributingFactors.push('critical_fractures')
        if (formData.rainfall > 150) contributingFactors.push('heavy_rainfall')
        if (formData.rainfall > 250) contributingFactors.push('extreme_rainfall')
        if (formData.seismic_activity > 4) contributingFactors.push('seismic_activity')
        if (formData.seismic_activity > 7) contributingFactors.push('major_earthquakes')
        if (formData.freeze_thaw_cycles > 20) contributingFactors.push('thermal_cycling')
        if (formData.temperature_variation > 20) contributingFactors.push('thermal_stress')
        if (formData.instability_index > 0.6) contributingFactors.push('geological_instability')
        if (formData.wetness_index > 0.7) contributingFactors.push('water_saturation')
        if (formData.precipitation_intensity > 15) contributingFactors.push('intense_precipitation')
        if (formData.wind_speed > 70) contributingFactors.push('high_winds')
        if (formData.elevation > 2500) contributingFactors.push('high_altitude')
        
        const results = {
          risk_score: riskScore,
          risk_level: riskLevel,
          confidence: Math.random() * 0.2 + 0.8, // 80-100% confidence for enhanced model
          recommendations: recommendations,
          contributing_factors: contributingFactors,
          probability_next_24h: riskScore * 0.85,
          probability_next_week: riskScore * 0.95,
          timestamp: new Date().toISOString()
        }
        
        console.log('✅ Risk assessment completed:', results)
        setRiskResults(results)
        
        // Add to history
        const newAssessment = {
          id: Date.now(),
          timestamp: new Date().toLocaleString(),
          risk_level: results.risk_level,
          risk_score: results.risk_score,
          confidence: results.confidence,
          recommendations: results.recommendations,
          data: formData
        }
        
        setAssessmentHistory(prev => [newAssessment, ...prev.slice(0, 9)]) // Keep last 10
        setLoading(false)
      }, 1200) // Simulate processing time
      
    } catch (err) {
      setError('Failed to assess risk. Please check your input values and try again.')
      console.error('Risk assessment error:', err)
      setLoading(false)
    }
  }
  
  const getRiskColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'low': return '#10b981'
      case 'medium': return '#f59e0b'
      case 'high': return '#ef4444'
      case 'critical': return '#dc2626'
      default: return '#6b7280'
    }
  }
  
  const getRiskIcon = (level) => {
    switch (level?.toLowerCase()) {
      case 'low': return <CheckIcon sx={{ color: '#10b981' }} />
      case 'medium': return <WarningIcon sx={{ color: '#f59e0b' }} />
      case 'high': return <ErrorIcon sx={{ color: '#ef4444' }} />
      case 'critical': return <ErrorIcon sx={{ color: '#dc2626' }} />
      default: return <AssessmentIcon sx={{ color: '#6b7280' }} />
    }
  }
  
  const resetForm = () => {
    setFormData({
      slope: 30,
      elevation: 1000,
      fracture_density: 2.5,
      roughness: 0.5,
      slope_variability: 0.3,
      instability_index: 0.4,
      wetness_index: 0.2,
      month: new Date().getMonth() + 1,
      day_of_year: Math.floor((new Date() - new Date(new Date().getFullYear(), 0, 0)) / 86400000),
      season: Math.floor((new Date().getMonth()) / 3),
      rainfall: 50,
      temperature: 20,
      temperature_variation: 5,
      freeze_thaw_cycles: 10,
      seismic_activity: 2,
      wind_speed: 10,
      precipitation_intensity: 5,
      humidity: 60,
      risk_score: 0.0
    })
    setRiskResults(null)
    setError(null)
  }
  
  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 700, mb: 1 }}>
          Risk Assessment
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Evaluate environmental conditions to predict rockfall risk probability
        </Typography>
      </Box>
      
      <Grid container spacing={3}>
        {/* Input Form */}
        <Grid item xs={12} lg={8}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Typography variant="h6" component="div" sx={{ mb: 3 }}>
                  Environmental Parameters
                </Typography>
                
                <Grid container spacing={3}>
                  {/* Geological Factors */}
                  <Grid item xs={12}>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <TerrainIcon sx={{ mr: 1, color: '#8b5cf6' }} />
                        Geological Factors
                      </Typography>
                      <Divider sx={{ borderColor: '#334155' }} />
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom>
                      Slope: {formData.slope}°
                    </Typography>
                    <Slider
                      value={formData.slope}
                      onChange={(e, value) => handleInputChange('slope', value)}
                      min={0}
                      max={90}
                      valueLabelDisplay="auto"
                      sx={{ color: '#3b82f6' }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom>
                      Elevation: {formData.elevation}m
                    </Typography>
                    <Slider
                      value={formData.elevation}
                      onChange={(e, value) => handleInputChange('elevation', value)}
                      min={0}
                      max={5000}
                      valueLabelDisplay="auto"
                      sx={{ color: '#3b82f6' }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Fracture Density (per m²)"
                      type="number"
                      value={formData.fracture_density}
                      onChange={(e) => handleInputChange('fracture_density', parseFloat(e.target.value) || 0)}
                      inputProps={{ step: 0.1, min: 0, max: 10 }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom>
                      Surface Roughness: {formData.roughness}
                    </Typography>
                    <Slider
                      value={formData.roughness}
                      onChange={(e, value) => handleInputChange('roughness', value)}
                      min={0}
                      max={1}
                      step={0.1}
                      valueLabelDisplay="auto"
                      sx={{ color: '#3b82f6' }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom>
                      Slope Variability: {formData.slope_variability}
                    </Typography>
                    <Slider
                      value={formData.slope_variability}
                      onChange={(e, value) => handleInputChange('slope_variability', value)}
                      min={0}
                      max={1}
                      step={0.1}
                      valueLabelDisplay="auto"
                      sx={{ color: '#3b82f6' }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom>
                      Instability Index: {formData.instability_index}
                    </Typography>
                    <Slider
                      value={formData.instability_index}
                      onChange={(e, value) => handleInputChange('instability_index', value)}
                      min={0}
                      max={1}
                      step={0.1}
                      valueLabelDisplay="auto"
                      sx={{ color: '#3b82f6' }}
                    />
                  </Grid>
                  
                  {/* Environmental Factors */}
                  <Grid item xs={12}>
                    <Box sx={{ mb: 2, mt: 3 }}>
                      <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <WaterIcon sx={{ mr: 1, color: '#06b6d4' }} />
                        Environmental Factors
                      </Typography>
                      <Divider sx={{ borderColor: '#334155' }} />
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Rainfall (mm)"
                      type="number"
                      value={formData.rainfall}
                      onChange={(e) => handleInputChange('rainfall', parseFloat(e.target.value) || 0)}
                      inputProps={{ min: 0, max: 500 }}
                      InputProps={{
                        endAdornment: <WaterIcon sx={{ color: '#06b6d4' }} />
                      }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Temperature (°C)"
                      type="number"
                      value={formData.temperature}
                      onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value) || 0)}
                      inputProps={{ min: -50, max: 50 }}
                      InputProps={{
                        endAdornment: <TempIcon sx={{ color: '#f59e0b' }} />
                      }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Temperature Variation (°C)"
                      type="number"
                      value={formData.temperature_variation}
                      onChange={(e) => handleInputChange('temperature_variation', parseFloat(e.target.value) || 0)}
                      inputProps={{ min: 0, max: 50 }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Humidity (%)"
                      type="number"
                      value={formData.humidity}
                      onChange={(e) => handleInputChange('humidity', parseFloat(e.target.value) || 0)}
                      inputProps={{ min: 0, max: 100 }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom>
                      Wetness Index: {formData.wetness_index}
                    </Typography>
                    <Slider
                      value={formData.wetness_index}
                      onChange={(e, value) => handleInputChange('wetness_index', value)}
                      min={0}
                      max={1}
                      step={0.1}
                      valueLabelDisplay="auto"
                      sx={{ color: '#06b6d4' }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Precipitation Intensity (mm/h)"
                      type="number"
                      value={formData.precipitation_intensity}
                      onChange={(e) => handleInputChange('precipitation_intensity', parseFloat(e.target.value) || 0)}
                      inputProps={{ min: 0, max: 100 }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Freeze-Thaw Cycles"
                      type="number"
                      value={formData.freeze_thaw_cycles}
                      onChange={(e) => handleInputChange('freeze_thaw_cycles', parseFloat(e.target.value) || 0)}
                      inputProps={{ min: 0, max: 50 }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Wind Speed (km/h)"
                      type="number"
                      value={formData.wind_speed}
                      onChange={(e) => handleInputChange('wind_speed', parseFloat(e.target.value) || 0)}
                      inputProps={{ min: 0, max: 200 }}
                      InputProps={{
                        endAdornment: <WindIcon sx={{ color: '#64748b' }} />
                      }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Seismic Activity (Magnitude)"
                      type="number"
                      value={formData.seismic_activity}
                      onChange={(e) => handleInputChange('seismic_activity', parseFloat(e.target.value) || 0)}
                      inputProps={{ min: 0, max: 10, step: 0.1 }}
                    />
                  </Grid>
                  
                  {/* Temporal Factors */}
                  <Grid item xs={12}>
                    <Box sx={{ mb: 2, mt: 3 }}>
                      <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <SpeedIcon sx={{ mr: 1, color: '#8b5cf6' }} />
                        Temporal Factors
                      </Typography>
                      <Divider sx={{ borderColor: '#334155' }} />
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      fullWidth
                      label="Month"
                      type="number"
                      value={formData.month}
                      onChange={(e) => handleInputChange('month', parseFloat(e.target.value) || 1)}
                      inputProps={{ min: 1, max: 12 }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      fullWidth
                      label="Day of Year"
                      type="number"
                      value={formData.day_of_year}
                      onChange={(e) => handleInputChange('day_of_year', parseFloat(e.target.value) || 1)}
                      inputProps={{ min: 1, max: 366 }}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <FormControl fullWidth>
                      <InputLabel>Season</InputLabel>
                      <Select
                        value={formData.season}
                        onChange={(e) => handleInputChange('season', e.target.value)}
                        label="Season"
                      >
                        <MenuItem value={0}>Spring</MenuItem>
                        <MenuItem value={1}>Summer</MenuItem>
                        <MenuItem value={2}>Autumn</MenuItem>
                        <MenuItem value={3}>Winter</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  
                  {/* Action Buttons */}
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', gap: 2, mt: 3 }}>
                      <Button
                        variant="contained"
                        onClick={handleAssessment}
                        disabled={loading}
                        startIcon={loading ? <CircularProgress size={20} /> : <AssessmentIcon />}
                        sx={{ flex: 1 }}
                      >
                        {loading ? 'Assessing...' : 'Assess Risk'}
                      </Button>
                      <Button
                        variant="outlined"
                        onClick={resetForm}
                        disabled={loading}
                      >
                        Reset
                      </Button>
                    </Box>
                  </Grid>
                </Grid>
                
                {/* Error Display */}
                {error && (
                  <Alert severity="error" sx={{ mt: 3 }}>
                    {error}
                  </Alert>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        {/* Results Section */}
        <Grid item xs={12} lg={4}>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="glass-card" sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" component="div" sx={{ mb: 3 }}>
                  Risk Assessment Results
                </Typography>
                
                {riskResults ? (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                  >
                    {/* Risk Level */}
                    <Paper 
                      sx={{ 
                        p: 3, 
                        textAlign: 'center', 
                        mb: 3,
                        backgroundColor: `${getRiskColor(riskResults.risk_level)}20`,
                        border: `2px solid ${getRiskColor(riskResults.risk_level)}`
                      }}
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
                        {getRiskIcon(riskResults.risk_level)}
                      </Box>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: getRiskColor(riskResults.risk_level), mb: 1 }}>
                        {riskResults.risk_level?.toUpperCase()}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Risk Level
                      </Typography>
                    </Paper>
                    
                    {/* Probability and Confidence */}
                    <Grid container spacing={2} sx={{ mb: 3 }}>
                      <Grid item xs={6}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h5" sx={{ fontWeight: 600, color: '#3b82f6' }}>
                            {(riskResults.risk_score * 100).toFixed(1)}%
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Probability
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h5" sx={{ fontWeight: 600, color: '#10b981' }}>
                            {(riskResults.confidence * 100).toFixed(1)}%
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Confidence
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                    
                    {/* Contributing Factors */}
                    {riskResults.contributing_factors && riskResults.contributing_factors.length > 0 && (
                      <Box>
                        <Typography variant="subtitle2" sx={{ mb: 2 }}>
                          Key Risk Factors:
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                          {riskResults.contributing_factors.map((factor, index) => (
                            <Chip
                              key={index}
                              label={factor.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              size="small"
                              sx={{
                                backgroundColor: getRiskColor(riskResults.risk_level),
                                color: 'white'
                              }}
                            />
                          ))}
                        </Box>
                      </Box>
                    )}
                  </motion.div>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <AssessmentIcon sx={{ fontSize: 64, color: '#64748b', mb: 2 }} />
                    <Typography variant="body1" color="text.secondary">
                      Configure parameters and click "Assess Risk" to see results
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        {/* Assessment History */}
        {assessmentHistory.length > 0 && (
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card className="glass-card">
                <CardContent>
                  <Typography variant="h6" component="div" sx={{ mb: 3 }}>
                    Assessment History
                  </Typography>
                  
                  <List>
                    {assessmentHistory.map((assessment, index) => (
                      <React.Fragment key={assessment.id}>
                        <ListItem>
                          <ListItemIcon>
                            {getRiskIcon(assessment.risk_level)}
                          </ListItemIcon>
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                <Typography variant="subtitle1">
                                  {assessment.risk_level?.toUpperCase()} Risk
                                </Typography>
                                <Chip
                                  label={`${(assessment.risk_score * 100).toFixed(1)}%`}
                                  size="small"
                                  sx={{
                                    backgroundColor: getRiskColor(assessment.risk_level),
                                    color: 'white'
                                  }}
                                />
                              </Box>
                            }
                            secondary={
                              <Box sx={{ mt: 1 }}>
                                <Typography variant="body2" color="text.secondary">
                                  {assessment.timestamp}
                                </Typography>
                                <LinearProgress
                                  variant="determinate"
                                  value={assessment.confidence * 100}
                                  sx={{ mt: 1, height: 4, borderRadius: 2 }}
                                />
                                <Typography variant="caption" color="text.secondary">
                                  Confidence: {(assessment.confidence * 100).toFixed(1)}%
                                </Typography>
                              </Box>
                            }
                          />
                        </ListItem>
                        {index < assessmentHistory.length - 1 && (
                          <Divider variant="inset" component="li" sx={{ borderColor: '#334155' }} />
                        )}
                      </React.Fragment>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        )}
      </Grid>
    </Box>
  )
}

export default RiskAssessment