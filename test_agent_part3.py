import numpy as np, torch
from pathlib import Path
from agent.forecaster.training_data import DataCollector, SEQ_LEN, N_FEATURES
from agent.forecaster.grad_forecaster import GradForecaster, ForecasterConfig
from agent.forecaster.failure_predictor import FailurePredictor
from agent.observer.rolling_window import RollingWindow, MetricSnapshot

print("=== FRAMEWORM-AGENT Part 3 Smoke Test ===\n")

print("1. DataCollector (synthetic fallback)...")
dataset = DataCollector(experiments_dir=Path("experiments")).collect()
print(f"   {len(dataset)} samples collected")
assert len(dataset) > 0
X, y = dataset.to_arrays()
assert X.ndim == 3 and y.ndim == 3
print(f"   X: {X.shape}, y: {y.shape} ✓")

print("\n2. GradForecaster architecture...")
config = ForecasterConfig(max_epochs=3, batch_size=16)
model = GradForecaster(config=config)
x_test = torch.randn(4, SEQ_LEN, N_FEATURES)
out = model(x_test)
assert out.shape == (4, 6, 3)
assert 0.0 <= out.min() and out.max() <= 1.0
print(f"   Output: {out.shape}, range [{out.min():.3f}, {out.max():.3f}] ✓")

print("\n3. model.fit() — 3 epochs...")
history = model.fit(dataset)
assert "val_loss" in history and len(history["val_loss"]) > 0
print(f"   Final val loss: {history['val_loss'][-1]:.4f} ✓")

print("\n4. model.predict()...")
fake_window = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
probs = model.predict(fake_window)
assert probs.shape == (6, 3)
print(f"   Probs shape: {probs.shape}, max: {probs.max():.3f} ✓")

print("\n5. Save / load roundtrip...")
p = Path("/tmp/test_fw_forecaster.pt")
model.save(p)
loaded = GradForecaster.load(p)
assert np.allclose(model.predict(fake_window), loaded.predict(fake_window), atol=1e-5)
print("   Roundtrip ✓")

print("\n6. FailurePredictor.tick()...")
predictor = FailurePredictor(model=loaded, confidence_threshold=0.01, run_every=1)
window = RollingWindow(size=200)
for i in range(120):
    window.push(MetricSnapshot(step=i, loss=1.0-i*0.005+np.random.normal(0,0.02), grad_norm=2.0, lr=0.0002))
result = predictor.tick(window=window, current_step=120, control=None)
assert result is not None
print(f"   {result.summary()} ✓")

print("\n7. load_or_init fallback...")
m = GradForecaster.load_or_init(Path("/tmp/nonexistent.pt"))
assert not m.is_ready
print("   Untrained fallback ✓")

print("\n✓ All Part 3 tests passed.")