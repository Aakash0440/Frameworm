"""
Smoke test for DEPLOY steps 1–5.
Run with: python test_deploy_steps1_5.py
No pytest required. No GPU required.
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m~\033[0m"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {name}" + (f"  —  {detail}" if detail else ""))
    results.append(condition)


def skip(name, reason=""):
    print(f"  {SKIP} {name}  (skipped: {reason})")
    results.append(True)


print("\n═══════════════════════════════════════════════")
print("  FRAMEWORM DEPLOY — Steps 1–5 smoke test")
print("═══════════════════════════════════════════════\n")


# ── Test 1: ModelExporter — signature lookup ──────────────────────────────
print("1. ModelExporter — architecture signatures")
from deploy.core.model_exporter import (FRAMEWORM_MODEL_SIGNATURES,
                                        ModelExporter)

exporter = ModelExporter()
check("VAE signature exists", "VAE" in FRAMEWORM_MODEL_SIGNATURES)
check("DCGAN signature exists", "DCGAN" in FRAMEWORM_MODEL_SIGNATURES)
check("DDPM signature exists", "DDPM" in FRAMEWORM_MODEL_SIGNATURES)
check("ViTGAN signature exists", "ViTGAN" in FRAMEWORM_MODEL_SIGNATURES)
check("CFG_DDPM has 3 inputs", len(FRAMEWORM_MODEL_SIGNATURES["CFG_DDPM"]["input_names"]) == 3)

sig = exporter._get_signature("DCGAN")
check("DCGAN input shape correct", sig["input_shape"] == (1, 100, 1, 1))

unknown_sig = exporter._get_signature("UNKNOWN_MODEL")
check("unknown model gets generic sig", unknown_sig["input_names"] == ["input"])


# ── Test 2: ModelExporter — dummy input creation ──────────────────────────
print("\n2. ModelExporter — dummy inputs")
try:
    import torch

    sig_ddpm = FRAMEWORM_MODEL_SIGNATURES["DDPM"]
    inputs = exporter._make_dummy_inputs(sig_ddpm, "DDPM")
    check("DDPM dummy is tuple", isinstance(inputs, tuple))
    check("DDPM dummy has 2 elements", len(inputs) == 2)
    check("DDPM noise shape correct", inputs[0].shape == (1, 3, 32, 32))

    sig_cfg = FRAMEWORM_MODEL_SIGNATURES["CFG_DDPM"]
    inputs3 = exporter._make_dummy_inputs(sig_cfg, "CFG_DDPM")
    check("CFG_DDPM dummy has 3 elements", len(inputs3) == 3)

    sig_vae = FRAMEWORM_MODEL_SIGNATURES["VAE"]
    inputs_v = exporter._make_dummy_inputs(sig_vae, "VAE")
    check("VAE dummy is tensor", isinstance(inputs_v, torch.Tensor))
    check("VAE dummy shape correct", inputs_v.shape == (1, 3, 64, 64))
except ImportError:
    skip("all dummy input tests", "torch not installed")


# ── Test 3: ModelExporter — TorchScript export with tiny model ───────────
print("\n3. ModelExporter — TorchScript export (tiny model)")
try:
    import torch
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        # Save a fake checkpoint
        model = TinyModel()
        ckpt_path = os.path.join(d, "tiny.pt")
        torch.save(
            {
                "model_class": "TinyModel",
                "model_state_dict": model.state_dict(),
            },
            ckpt_path,
        )

        # Patch exporter to not use model registry
        exp = ModelExporter(exports_dir=d)
        sig = exp._get_signature("TinyModel")  # gets generic sig
        # Override with correct shape for TinyModel
        import torch

        dummy = torch.randn(1, 10)
        ts_path = os.path.join(d, "tiny.pt_ts")
        exp._export_torchscript(model, dummy, ts_path, "TinyModel")
        check("TorchScript file created", os.path.exists(ts_path))
        loaded = torch.jit.load(ts_path)
        out = loaded(dummy)
        check("TorchScript runs correctly", out.shape == (1, 5))
except ImportError:
    skip("TorchScript export test", "torch not installed")
except Exception as e:
    check("TorchScript export", False, str(e))


# ── Test 4: ExportManifest serialisation ─────────────────────────────────
print("\n4. ExportManifest — serialisation")
from deploy.core.model_exporter import ExportManifest

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    manifest = ExportManifest(
        model_name="test_model",
        model_class="DCGAN",
        checkpoint_path="/fake/ckpt.pt",
        checkpoint_hash="abc123",
        export_dir=d,
        torchscript_path=os.path.join(d, "model.pt"),
        onnx_path=os.path.join(d, "model.onnx"),
        quantized_path=None,
        input_shape=[1, 100, 1, 1],
        input_names=["noise"],
        output_names=["generated_image"],
        export_formats=["torchscript", "onnx"],
        quantized=False,
        exported_at="2024-01-01T00:00:00",
        metadata={"tag": "test"},
    )
    path = os.path.join(d, "manifest.json")
    manifest.save(path)
    check("manifest file created", os.path.exists(path))
    restored = ExportManifest.load(path)
    check("model_name round-trips", restored.model_name == "test_model")
    check("model_class round-trips", restored.model_class == "DCGAN")
    check("input_shape round-trips", restored.input_shape == [1, 100, 1, 1])
    check("metadata round-trips", restored.metadata == {"tag": "test"})


# ── Test 5: ModelRegistry — register + transition ────────────────────────
print("\n5. ModelRegistry — lifecycle")
from deploy.core.registry import ModelRegistry, ModelStage

# Use ignore_cleanup_errors=True so Windows PermissionErrors on cleanup are
# swallowed instead of crashing the test run.  The registry itself now uses
# journal_mode=DELETE which prevents SQLite WAL sidecar files, so in practice
# cleanup succeeds — but the flag is a belt-and-suspenders safety net.
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    db_path = os.path.join(d, "registry.db")
    registry = ModelRegistry(db_path=db_path)

    # Register
    record = registry.register(
        model_name="dcgan_v1",
        manifest=manifest,
        notes="smoke test",
    )
    check("record created", record is not None)
    check("starts in DEV stage", record.stage == "dev")
    check("version is 1.0.0", record.version == "1.0.0")
    check("model_name correct", record.model_name == "dcgan_v1")

    # Transition DEV → STAGING
    record = registry.transition(record.id, ModelStage.STAGING)
    check("transitioned to STAGING", record.stage == "staging")

    # Transition STAGING → PRODUCTION
    record = registry.transition(record.id, ModelStage.PRODUCTION)
    check("transitioned to PRODUCTION", record.stage == "production")
    check("deployed_at set", record.deployed_at is not None)

    # get_production
    prod = registry.get_production("dcgan_v1")
    check("get_production returns record", prod is not None)
    check("get_production is correct", prod.id == record.id)

    # Register v2 — should archive v1
    record2 = registry.register("dcgan_v1", manifest)
    check("v2 version is 1.0.1", record2.version == "1.0.1")
    registry.transition(record2.id, ModelStage.STAGING)
    registry.transition(record2.id, ModelStage.PRODUCTION)
    old = registry.get(record.id)
    check("v1 archived after v2 promoted", old.stage == "archived")
    check("v2 is now production", registry.get_production("dcgan_v1").version == "1.0.1")

    # Invalid transition
    try:
        registry.transition(record2.id, ModelStage.DEV)
        check("invalid transition raises error", False)
    except ValueError:
        check("invalid transition raises ValueError", True)

    # History
    history = registry.history("dcgan_v1")
    check("history has 2 records", len(history) == 2)

    # list_models
    models = registry.list_models()
    check("list_models finds dcgan_v1", "dcgan_v1" in models)

    # Explicitly close / checkpoint before TemporaryDirectory cleanup.
    # This is the primary fix for the Windows WinError 32 PermissionError:
    # we flush all pending SQLite writes and release the file handle before
    # the OS tries to delete the temp folder.
    registry.close()


# ── Test 6: LatencyTracker ────────────────────────────────────────────────
print("\n6. LatencyTracker — p50/p95/p99")
import time

from deploy.core.latency_tracker import LatencyTracker, get_tracker

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    tracker = LatencyTracker("test_model", logs_dir=d)

    # Simulate 200 requests (100ms ± noise)
    for i in range(200):
        start = tracker.start_request()
        time.sleep(0.0005)  # 0.5ms simulated work
        tracker.end_request(start, success=(i % 20 != 0))  # 5% error rate

    snap = tracker.snapshot()
    check("snapshot not None", snap is not None)
    check("n_requests == 200", snap.n_requests == 200)
    check("p50 > 0", snap.p50_ms > 0)
    check("p95 >= p50", snap.p95_ms >= snap.p50_ms)
    check("p99 >= p95", snap.p99_ms >= snap.p95_ms)
    check("error_rate ~5%", abs(snap.error_rate - 0.05) < 0.02)
    check("breaches_threshold ok at 500ms", snap.breaches_threshold(200, 500) == "ok")
    check("breaches_threshold crit logic", snap.breaches_threshold(0, 0) == "critical")

    # get_tracker singleton
    t1 = get_tracker("singleton_model")
    t2 = get_tracker("singleton_model")
    check("get_tracker returns singleton", t1 is t2)


# ── Summary ───────────────────────────────────────────────────────────────
print("\n═══════════════════════════════════════════════")
passed = sum(results)
total = len(results)
colour = "\033[92m" if passed == total else "\033[91m"
print(f"  {colour}{passed}/{total} passed\033[0m")
print("═══════════════════════════════════════════════\n")
if passed < total:
    sys.exit(1)
