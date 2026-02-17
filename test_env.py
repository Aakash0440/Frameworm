import os
import yaml
from core import Config

# Set environment variable
os.environ["DATA_ROOT"] = "/path/to/data"
os.environ["EXPERIMENT_NAME"] = "test_exp"

# Create config with env vars
config_content = {
    "paths": {"data_dir": "${DATA_ROOT}/images", "output_dir": "./outputs/${EXPERIMENT_NAME}"},
    "experiment": {"name": "${EXPERIMENT_NAME}"},
}

with open("test_env.yaml", "w") as f:
    yaml.dump(config_content, f)

# Load config
cfg = Config("test_env.yaml")

# Check interpolation
assert cfg.paths.data_dir == "/path/to/data/images"
assert cfg.paths.output_dir == "./outputs/test_exp"
assert cfg.experiment.name == "test_exp"

print("✓ Environment variable interpolation works!")
print(f"  data_dir: {cfg.paths.data_dir}")
print(f"  output_dir: {cfg.paths.output_dir}")

# Clean up
os.remove("test_env.yaml")
del os.environ["DATA_ROOT"]
del os.environ["EXPERIMENT_NAME"]
