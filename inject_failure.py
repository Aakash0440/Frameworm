import json, os
from pathlib import Path

metrics_path = Path(os.environ.get('TEMP', '/tmp')) / 'frameworm_agent_metrics.json'

metrics_path.write_text(json.dumps({
    "step": 999,
    "loss": 847.3,
    "grad_norm": 94.2,
    "lr": 0.0001,
    "epoch": 50
}))
print("Injected loss explosion")
