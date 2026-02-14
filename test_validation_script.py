from core import Config
from pydantic import BaseModel, Field, validator
import yaml
import os

# Define schema
class TrainingConfig(BaseModel):
    epochs: int = Field(gt=0, description="Number of training epochs")
    batch_size: int = Field(gt=0, le=512)
    device: str

    @validator("device")
    def validate_device(cls, v):
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"device must be cpu, cuda, or mps, got {v}")
        return v


# ---- VALID CONFIG TEST ----
cfg = Config("configs/base.yaml")

validated_training = cfg.training.validate(TrainingConfig)


try:
    validated = cfg.training.validate(TrainingConfig)
    print("✓ Validation passed")
    print(f"  epochs: {validated.epochs}")
    print(f"  batch_size: {validated.batch_size}")
    print(f"  device: {validated.device}")
except Exception as e:
    print(f"✗ Validation failed: {e}")


# ---- INVALID CONFIG TEST ----
with open("test_invalid.yaml", "w") as f:
    yaml.dump({"epochs": -1, "batch_size": 32, "device": "cuda"}, f)

cfg_invalid = Config("test_invalid.yaml")

try:
    cfg_invalid.validate(TrainingConfig)
    print("✗ Should have failed validation")
except Exception as e:
    print("✓ Correctly rejected invalid config")
    print(f"  Error: {str(e)[:100]}")

os.remove("test_invalid.yaml")
