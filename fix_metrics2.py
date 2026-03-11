import pathlib
f = pathlib.Path('training/metrics.py')
c = f.read_text(encoding='utf-8')
old = '        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])'
new = '        metrics_str = " | ".join([f"{k}: {float(v):.4f}" for k, v in metrics.items() if hasattr(v, "__float__") and (not hasattr(v, "numel") or v.numel() == 1)])'
if old in c:
    c = c.replace(old, new)
    f.write_text(c, encoding='utf-8')
    print("patched OK")
else:
    print("pattern not found")
