# Web UI Dashboard Design

## Architecture
Frontend (React)
↓
FastAPI Backend
↓
FRAMEWORM Core

## Features

### 1. Dashboard Home
- Active experiments
- Recent activity
- System status
- Quick actions

### 2. Experiments
- List all experiments
- Filter & search
- Compare experiments
- Metrics visualization

### 3. Models
- Model registry
- Export models
- Deploy models
- Performance metrics

### 4. Training
- Start new training
- Monitor progress
- Real-time logs
- Resource usage

### 5. Deployment
- Model serving status
- API endpoints
- Performance metrics
- Health checks

## Tech Stack

**Backend:**
- FastAPI
- SQLite (experiment DB)
- WebSockets (real-time)

**Frontend:**
- React
- Recharts (plots)
- Tailwind CSS
- Axios

## API Endpoints
GET  /api/experiments          # List experiments
GET  /api/experiments/{id}     # Get experiment
POST /api/experiments          # Create experiment
GET  /api/experiments/{id}/metrics  # Get metrics
WS   /api/training/stream      # Real-time training updates
GET  /api/models               # List models
POST /api/models/export        # Export model
POST /api/models/deploy        # Deploy model
GET  /api/system/status        # System status
GET  /api/system/resources     # GPU/CPU usage