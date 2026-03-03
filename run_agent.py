from agent.react.agent import FramewormAgent
import time, os, json
from pathlib import Path

agent = FramewormAgent.from_config('configs/base.yaml', run_id=None)
agent.start()
print('Agent monitoring...')

metrics_path = Path(os.environ.get('TEMP', '/tmp')) / 'frameworm_agent_metrics.json'

while True:
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text())
        print(f"Step {data['step']} | loss: {data['loss']:.4f}")
    else:
        print('Waiting for training metrics...')
    time.sleep(10)
