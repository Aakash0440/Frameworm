# Frequently Asked Questions

---

## General

**Q: What makes FRAMEWORM different?**

A: FRAMEWORM is all-in-one. Training, tracking, search, deployment - everything integrated.

**Q: Is FRAMEWORM production-ready?**

A: Yes! Used in production at several companies.

**Q: Does it support distributed training?**

A: Yes, both DataParallel and DistributedDataParallel.

---

## Installation

**Q: Which Python versions are supported?**

A: Python 3.8+ (3.10 recommended)

**Q: Does it work on Windows?**

A: Yes, but Linux is recommended for production.

**Q: Can I use it without GPUs?**

A: Yes, CPU training works fine (just slower).

---

## Training

**Q: How do I resume training?**

A: Use `--resume checkpoint.pt` with CLI or:
```python
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**Q: Why is training slow?**

A: Common causes:
- CPU instead of GPU
- Small batch size
- Data loading bottleneck
- No mixed precision

---

## Deployment

**Q: Which export format should I use?**

A: 
- **TorchScript** - PyTorch native, C++ deployable
- **ONNX** - Framework agnostic, TensorRT support
- **Quantized** - 4x smaller, 2-3x faster

**Q: How do I deploy to production?**

A: See [Production Deployment Guide](examples/production-deployment.md)

---

## Troubleshooting

**Q: CUDA out of memory**

A: Reduce batch size or enable gradient accumulation.

**Q: Model not converging**

A: Try:
- Lower learning rate
- Different optimizer
- Hyperparameter search

**Q: Import errors**

A: Reinstall: `pip install --force-reinstall frameworm`