
"""
Web UI API backend.
"""

from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from pathlib import Path
import json
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from experiment import ExperimentManager
from deployment import ModelExporter


# Pydantic models
class ExperimentCreate(BaseModel):
    name: str
    config: Dict[str, Any]
    description: Optional[str] = None
    tags: List[str] = []


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    status: str
    created_at: str
    metrics: Optional[Dict[str, float]] = None


class ModelExportRequest(BaseModel):
    checkpoint_path: str
    format: str = "torchscript"
    output_dir: str = "exported_models"


class TrainingStatus(BaseModel):
    status: str
    epoch: int
    total_epochs: int
    metrics: Dict[str, float]
    progress: float


# Create API
app = FastAPI(
    title="FRAMEWORM Dashboard",
    description="Web UI for experiment tracking and model management",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
experiment_manager = ExperimentManager("experiments")
active_trainings: Dict[str, Dict] = {}


# Root
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FRAMEWORM Dashboard API",
        "version": "1.0.0",
        "docs": "/docs"
    }


# Experiments API
@app.get("/api/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    status: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    limit: int = 100
):
    """
    List all experiments.
    
    Query parameters:
    - status: Filter by status (running, completed, failed)
    - tags: Filter by tags
    - limit: Maximum results
    """
    df = experiment_manager.list_experiments(status=status, tags=tags, limit=limit)
    
    experiments = []
    for _, row in df.iterrows():
        exp = ExperimentResponse(
            experiment_id=row['experiment_id'],
            name=row['name'],
            status=row['status'],
            created_at=str(row['created_at'])
        )
        experiments.append(exp)
    
    return experiments


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details"""
    try:
        exp = experiment_manager.get_experiment(experiment_id)
        return exp
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/experiments", response_model=ExperimentResponse)
async def create_experiment(experiment: ExperimentCreate):
    """Create new experiment"""
    from frameworm.experiment import Experiment
    
    exp = Experiment(
        name=experiment.name,
        config=experiment.config,
        description=experiment.description,
        tags=experiment.tags
    )
    exp.start()
    
    return ExperimentResponse(
        experiment_id=exp.experiment_id,
        name=exp.name,
        status=exp.status,
        created_at=str(datetime.now())
    )


@app.get("/api/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    """Get experiment metrics"""
    try:
        metrics = experiment_manager.get_experiment(experiment_id)['metrics_summary']
        return metrics
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/experiments/{experiment_id}/compare")
async def compare_experiments(experiment_ids: List[str]):
    """Compare multiple experiments"""
    try:
        comparison = experiment_manager.compare_experiments(experiment_ids)
        return comparison.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Models API
@app.get("/api/models")
async def list_models():
    """List available models"""
    # Scan for model checkpoints
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        return []
    
    models = []
    for path in checkpoint_dir.glob("**/*.pt"):
        models.append({
            "name": path.stem,
            "path": str(path),
            "size_mb": path.stat().st_size / (1024 * 1024),
            "modified": path.stat().st_mtime
        })
    
    return models

static_dir = Path(__file__).parent / 'static'
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    
@app.post("/api/models/export")
async def export_model(request: ModelExportRequest):
    """Export model to deployment format"""
    import torch
    
    try:
        # Load checkpoint
        checkpoint = torch.load(request.checkpoint_path)
        model = checkpoint.get('model')
        
        if model is None:
            raise ValueError("Model not found in checkpoint")
        
        # Create exporter
        example_input = torch.randn(1, 3, 64, 64)  # Default size
        exporter = ModelExporter(model, example_input)
        
        # Export
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if request.format == "torchscript":
            output_path = output_dir / "model.pt"
            exporter.to_torchscript(str(output_path))
        elif request.format == "onnx":
            output_path = output_dir / "model.onnx"
            exporter.to_onnx(str(output_path))
        else:
            raise ValueError(f"Unknown format: {request.format}")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "format": request.format
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# System API
@app.get("/api/system/status")
async def get_system_status():
    """Get system status"""
    import torch
    import psutil
    
    status = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
    }
    
    if torch.cuda.is_available():
        status["gpu_count"] = torch.cuda.device_count()
        status["gpus"] = []
        
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            status["gpus"].append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": mem
            })
    else:
        status["gpu_count"] = 0
    
    return status


@app.get("/api/system/resources")
async def get_resource_usage():
    """Get current resource usage"""
    import torch
    import psutil
    
    resources = {
        "cpu": psutil.cpu_percent(percpu=True),
        "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
    }
    
    if torch.cuda.is_available():
        resources["gpus"] = []
        for i in range(torch.cuda.device_count()):
            from frameworm.distributed import get_gpu_memory_usage
            mem = get_gpu_memory_usage(i)
            resources["gpus"].append({
                "id": i,
                "memory_allocated_gb": mem['allocated_gb'],
                "memory_reserved_gb": mem['reserved_gb']
            })
    
    return resources


# Training WebSocket
@app.websocket("/api/training/stream")
async def training_stream(websocket: WebSocket):
    """
    Real-time training updates via WebSocket.
    
    Sends updates on training progress, metrics, etc.
    """
    await websocket.accept()
    
    try:
        while True:
            # Send training status updates
            # In practice, this would read from active training sessions
            status = {
                "status": "training",
                "epoch": 10,
                "total_epochs": 100,
                "metrics": {
                    "loss": 0.5,
                    "val_loss": 0.6
                },
                "progress": 0.1
            }
            
            await websocket.send_json(status)
            
            # Wait for client message or timeout
            import asyncio
            await asyncio.sleep(1)
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Dashboard stats
@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    df = experiment_manager.list_experiments()
    
    stats = {
        "total_experiments": len(df),
        "running": len(df[df['status'] == 'running']),
        "completed": len(df[df['status'] == 'completed']),
        "failed": len(df[df['status'] == 'failed']),
        "recent": df.head(5).to_dict(orient='records')
    }
    
    return stats


# Run server
def run_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Run the dashboard server"""
    import uvicorn
    
    print(f"Starting FRAMEWORM Dashboard at http://{host}:{port}")
    print(f"API docs at http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()