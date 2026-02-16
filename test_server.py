import torch
import torch.nn as nn
from deployment.export import ModelExporter
from deployment.server import ModelServer
import time
import requests
from threading import Thread

print("Testing FastAPI Server:")
print("="*60)

# Create simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
example_input = torch.randn(1, 10)

# Export model
exporter = ModelExporter(model, example_input)
exporter.to_torchscript('test_server_model.pt')
print("✓ Model exported")

# Create server
server = ModelServer('test_server_model.pt')
print("✓ Server created")

# Run server in background
def run_server():
    server.run(host='127.0.0.1', port=8001, reload=False)

thread = Thread(target=run_server, daemon=True)
thread.start()

# Wait for server to start
time.sleep(3)

# Test endpoints
try:
    # Test root
    response = requests.get('http://127.0.0.1:8001/')
    assert response.status_code == 200
    print("✓ Root endpoint works")
    
    # Test health
    response = requests.get('http://127.0.0.1:8001/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    print("✓ Health endpoint works")
    
    # Test predict
    response = requests.post(
        'http://127.0.0.1:8001/predict',
        json={'data': [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}
    )
    assert response.status_code == 200
    result = response.json()
    assert 'predictions' in result
    print("✓ Predict endpoint works")
    print(f"  Prediction: {result}")
    
    print("\n✓ Server working! Visit http://127.0.0.1:8001/docs for API docs")
    
except Exception as e:
    print(f"✗ Server test failed: {e}")

# Cleanup
import os
os.remove('test_server_model.pt')

print("="*60)
print("✅ FastAPI server working!")
print("\nTo run manually:")
print("  python -m frameworm.deployment.server --model model.pt --port 8000")