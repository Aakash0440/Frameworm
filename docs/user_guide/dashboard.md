# Web Dashboard

## Overview

FRAMEWORM includes a web-based dashboard for experiment tracking, model management, and training monitoring.

## Launch Dashboard
```bash
frameworm dashboard --port 8080
```

Then open http://localhost:8080 in your browser.

## Features

### 1. Dashboard Home
- Overview of all experiments
- System resource usage
- Recent activity

### 2. Experiments
- List and filter experiments
- View experiment details
- Compare multiple experiments
- Visualize metrics

### 3. Models
- Model registry
- Export models (TorchScript, ONNX)
- Deploy models
- Model metadata

### 4. Training Monitor
- Start/stop training
- Real-time metrics
- Training logs
- Progress tracking

## API

The dashboard runs on FastAPI backend.

API documentation: http://localhost:8080/docs

### Key Endpoints
GET  /api/experiments           # List experiments
GET  /api/experiments/{id}      # Get experiment
POST /api/experiments           # Create experiment
GET  /api/models                # List models
POST /api/models/export         # Export model
GET  /api/system/status         # System status
WS   /api/training/stream       # Real-time training

## Development

### Frontend Development
```bash
cd frameworm/ui/frontend
npm install
npm start  # Development server on port 3000
```

### Backend Development
```bash
python -m frameworm.ui.api
```

### Build Production Bundle
```bash
./frameworm/ui/build_frontend.sh
```

## Configuration

Set environment variables:
```bash
export FRAMEWORM_EXPERIMENT_DIR=/path/to/experiments
export FRAMEWORM_CHECKPOINT_DIR=/path/to/checkpoints
```

## Screenshots

(Add screenshots of dashboard pages)

## Troubleshooting

### Port Already in Use
```bash
frameworm dashboard --port 8081
```

### API Connection Issues

Check that backend is running:
```bash
curl http://localhost:8080/api/system/status
```