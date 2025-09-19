# 🏔️ Rockfall Detection & Prediction System

A sophisticated React-based frontend application for real-time rockfall detection and geological risk assessment, designed for mining safety and geological monitoring applications.

![Dashboard Preview](rockfall_detection/frontend/public/rockfall-icon.svg)

## 🌟 Features

- **Real-time Risk Assessment** - Dynamic environmental monitoring with intelligent risk scoring
- **AI-Powered Rock Detection** - Computer vision system for automated rockfall identification
- **Live Monitoring Dashboard** - Multi-camera surveillance with real-time alerts
- **DEM Analysis** - Digital Elevation Model processing for terrain assessment
- **Interactive Notifications** - Smart alert system with risk-based escalation
- **Responsive Design** - Mobile-friendly interface with Material-UI components

## 🏗️ Architecture

This is a **standalone frontend application** that operates in showcase mode with sophisticated mock data simulation, designed to demonstrate the full capabilities of a rockfall detection system without requiring backend infrastructure.

### Tech Stack

- **Frontend**: React 18 + Vite
- **UI Framework**: Material-UI (MUI) v5
- **Charts**: Recharts for data visualization
- **Animations**: Framer Motion
- **Routing**: React Router v6
- **Notifications**: React Hot Toast

## 🚀 Quick Start

### Prerequisites

- Node.js >= 16.0.0
- npm or yarn package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SAURABHM6341/VOID-SIH25-GO5V2.git
   cd VOID-SIH25-GO5V2/rockfall_detection/frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   ```
   http://localhost:3000
   ```

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
npm test         # Run tests
```

## 📱 Application Modules

### 🎯 Dashboard
Central monitoring hub displaying:
- Real-time risk metrics and environmental data
- System status and model health
- Recent activity feed
- Risk trend analysis over 24 hours

### 📹 Live Monitoring
Multi-camera surveillance system featuring:
- Three camera feeds (East, West, North positions)
- Real-time detection overlays
- Alert management and response protocols

### 🗺️ DEM Analysis
Digital Elevation Model processing with:
- Interactive terrain maps for multiple mining sites
- Elevation profile analysis
- Slope stability assessments

### 🔍 Rock Detection
AI-powered detection interface offering:
- Upload and analyze custom images
- Real-time confidence scoring
- Bounding box visualization
- Detection statistics and metrics

### ⚠️ Risk Assessment
Comprehensive risk evaluation system:
- Multi-factor environmental analysis
- Historical risk trend visualization
- Predictive risk modeling
- Automated alert thresholds

## ⚙️ Configuration

The application uses a comprehensive configuration system via `src/config/index.js`. Key environment variables:

```bash
# Feature Toggles
VITE_ENABLE_LIVE_MONITORING=true
VITE_ENABLE_DEM_ANALYSIS=true
VITE_ENABLE_RISK_ASSESSMENT=true
VITE_ENABLE_DETECTION=true

# Risk System
VITE_HIGH_RISK_THRESHOLD=75
VITE_RISK_FORCE_HIGH_CHANCE=0.2
VITE_NOTIFICATION_COOLDOWN=120000

# Camera Configuration
VITE_CAMERA_ENABLED=true
VITE_CAMERA_REFRESH_RATE=30

# Performance
VITE_ENABLE_CACHE=true
VITE_AUTO_REFRESH_INTERVAL=5000
```

## 🎨 Design System

### Color Palette
- **Primary**: Deep blue (`#3b82f6`) for navigation and primary actions
- **Secondary**: Purple (`#8b5cf6`) for detection features
- **Success**: Green (`#10b981`) for system status
- **Warning**: Amber (`#f59e0b`) for medium risk alerts
- **Error**: Red (`#ef4444`) for high risk scenarios

### Theme
- **Dark Mode**: Default theme with dark backgrounds (`#0f172a`, `#1e293b`)
- **Typography**: Clean, modern font stack with clear hierarchy
- **Icons**: Material Design icons for consistency

## 🔄 Data Flow

### Risk Assessment Pipeline
1. **Environmental Data Collection** - Simulated sensor readings (rainfall, temperature, seismic activity)
2. **Risk Calculation** - Multi-factor algorithm combining environmental factors
3. **Alert Generation** - Threshold-based notification system
4. **Cross-Component Communication** - Props-based state sharing between Dashboard and App

### Detection Workflow
1. **Image Processing** - Static image analysis with pre-computed results
2. **Confidence Scoring** - ML model confidence simulation
3. **Bounding Box Rendering** - Visual overlay of detected objects
4. **Statistics Tracking** - Detection count and performance metrics

## 📊 Mock Data System

The application includes sophisticated mock data generation:

- **Realistic Environmental Simulation**: Weather patterns, seismic activity, geological factors
- **Detection Results**: Pre-computed ML detection outputs with bounding boxes
- **Risk Calculations**: Dynamic risk scoring with configurable high-risk scenarios
- **Historical Data**: 24-hour trend simulation for dashboards

## 🛠️ Development Guidelines

### Project Structure
```
src/
├── components/     # Reusable UI components
├── pages/         # Main application pages
├── config/        # Configuration and environment setup
├── assets/        # Static assets and images
└── utils/         # Utility functions and helpers
```

### Component Patterns
- **Page Components**: Follow common props interface (`systemStatus`, `connectionStatus`, `onRiskDataUpdate`)
- **State Management**: Centralized in App.jsx with props drilling
- **Styling**: Material-UI `sx` prop for component-level styling

### Adding New Features
1. Update navigation in `App.jsx`
2. Create page component following existing patterns
3. Add configuration flags in `src/config/index.js`
4. Update routing logic in `renderCurrentPage()`

## 🧪 Testing

The application includes test configuration for:
- **Jest**: Unit testing framework
- **ESLint**: Code quality and consistency
- **React Testing Library**: Component testing (setup ready)

## 📦 Building for Production

```bash
npm run build
```

Generates optimized production build in `dist/` directory with:
- Code splitting and tree shaking
- Source maps for debugging
- Asset optimization
- Environment variable injection

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the Smart India Hackathon 2025 submission by Team VOID-GO5V2.

## 🏆 Acknowledgments

- **Team VOID-GO5V2** - Smart India Hackathon 2025 participants
- **Material-UI** - Comprehensive React UI framework
- **Recharts** - Composable charting library
- **Framer Motion** - Production-ready motion library

---

**Note**: This is a frontend showcase application designed to demonstrate the full capabilities of a rockfall detection system. For production deployment, integrate with appropriate backend services for real-time data processing and ML model inference.