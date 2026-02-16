"""
FastAPI model serving.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
from PIL import Image
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    data: List[List[float]]


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[float]
    probabilities: Optional[List[List[float]]] = None


class ModelServer:
    """
    FastAPI-based model serving.
    
    Provides REST API for model inference.
    
    Args:
        model_path: Path to model file (.pt or .onnx)
        preprocessor: Optional preprocessing function
        
    Example:
        >>> server = ModelServer('model.pt')
        >>> server.run(host='0.0.0.0', port=8000)
    """
    
    def __init__(
        self,
        model_path: str,
        preprocessor: Optional[callable] = None,
        device: str = 'cpu'
    ):
        self.model_path = model_path
        self.preprocessor = preprocessor
        self.device = device
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="FRAMEWORM Model Server",
            description="Production model serving API",
            version="1.0.0"
        )
        
        # Add CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
        logger.info(f"✓ Model server initialized: {model_path}")
    
    def _load_model(self, model_path: str):
        """Load model from file"""
        if model_path.endswith('.pt'):
            # TorchScript
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
            logger.info("Loaded TorchScript model")
        elif model_path.endswith('.onnx'):
            # ONNX Runtime
            from frameworm.deployment.onnx_runtime import ONNXInferenceSession
            model = ONNXInferenceSession(model_path)
            logger.info("Loaded ONNX model")
        else:
            # Regular PyTorch
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            logger.info("Loaded PyTorch model")
        
        return model
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "FRAMEWORM Model Server",
                "model": self.model_path,
                "status": "ready"
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None
            }
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """
            Prediction endpoint for JSON data.
            
            Request:
                {
                    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                }
            
            Response:
                {
                    "predictions": [0.8, 0.2],
                    "probabilities": [[0.2, 0.8], [0.9, 0.1]]
                }
            """
            try:
                # Convert to tensor
                data = torch.tensor(request.data, dtype=torch.float32)
                
                # Inference
                with torch.no_grad():
                    if isinstance(self.model, torch.jit.ScriptModule):
                        output = self.model(data)
                    else:
                        # ONNX Runtime
                        output = self.model.run(data.numpy())
                        output = torch.tensor(output)
                
                # Process output
                if output.dim() == 2:
                    # Classification
                    probabilities = torch.softmax(output, dim=1).tolist()
                    predictions = output.argmax(dim=1).tolist()
                else:
                    # Regression
                    predictions = output.tolist()
                    probabilities = None
                
                return PredictionResponse(
                    predictions=predictions,
                    probabilities=probabilities
                )
            
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/image")
        async def predict_image(file: UploadFile = File(...)):
            """
            Prediction endpoint for images.
            
            Upload an image file and get predictions.
            """
            try:
                # Read image
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                
                # Preprocess
                if self.preprocessor:
                    tensor = self.preprocessor(image)
                else:
                    # Default: resize to 224x224, normalize
                    from torchvision import transforms
                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])
                    tensor = preprocess(image).unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    if isinstance(self.model, torch.jit.ScriptModule):
                        output = self.model(tensor)
                    else:
                        output = self.model.run(tensor.numpy())
                        output = torch.tensor(output)
                
                # Process output
                probabilities = torch.softmax(output, dim=1)[0]
                prediction = probabilities.argmax().item()
                confidence = probabilities.max().item()
                
                return JSONResponse({
                    "prediction": prediction,
                    "confidence": float(confidence),
                    "probabilities": probabilities.tolist()
                })
            
            except Exception as e:
                logger.error(f"Image prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/batch")
        async def predict_batch(request: PredictionRequest):
            """
            Batch prediction endpoint.
            
            More efficient for multiple samples.
            """
            try:
                # Convert to tensor
                data = torch.tensor(request.data, dtype=torch.float32)
                
                # Batch inference
                with torch.no_grad():
                    if isinstance(self.model, torch.jit.ScriptModule):
                        output = self.model(data)
                    else:
                        output = self.model.run(data.numpy())
                        output = torch.tensor(output)
                
                # Process output
                predictions = output.argmax(dim=1).tolist()
                probabilities = torch.softmax(output, dim=1).tolist()
                
                return PredictionResponse(
                    predictions=predictions,
                    probabilities=probabilities
                )
            
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/model/info")
        async def model_info():
            """Get model information"""
            return {
                "model_path": self.model_path,
                "device": self.device,
                "model_type": type(self.model).__name__
            }
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        workers: int = 1
    ):
        """
        Run the server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            reload: Auto-reload on code changes
            workers: Number of worker processes
        """
        import uvicorn
        
        logger.info(f"Starting server at http://{host}:{port}")
        logger.info(f"API docs at http://{host}:{port}/docs")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            workers=workers
        )


# Standalone server script
def create_server(model_path: str, host: str = "0.0.0.0", port: int = 8000):
    """
    Create and run model server.
    
    Usage:
        python -m frameworm.deployment.server --model model.pt --port 8000
    """
    server = ModelServer(model_path)
    server.run(host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FRAMEWORM Model Server")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    create_server(args.model, args.host, args.port)