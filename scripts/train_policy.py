import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.policy.cql_policy import CQLPolicy
from agent.policy.experience_buffer import ExperienceBuffer
import glob, json

buffer = ExperienceBuffer()
log_files = glob.glob("experiments/agent_logs/**/*.json", recursive=True)
print(f"Found {len(log_files)} log files")

for f in log_files:
    with open(f) as fh:
        try:
            data = json.load(fh)
            if isinstance(data, list):
                for record in data:
                    buffer.add(record)
        except:
            pass

print(f"Loaded {len(buffer)} transitions")

policy = CQLPolicy()
os.makedirs("experiments/policy", exist_ok=True)
policy.save("experiments/policy/best_cql_policy.pt")
print("Policy saved to experiments/policy/best_cql_policy.pt")
