import pathlib
f = pathlib.Path('training/metrics.py')
c = f.read_text(encoding='utf-8')
old = '            if hasattr(value, "item"):\n                value = value.item()\n            self.batch_metrics[name].append(value)'
new = '            if hasattr(value, "item"):\n                if value.numel() != 1:\n                    continue\n                value = value.item()\n            self.batch_metrics[name].append(value)'
if old in c:
    c = c.replace(old, new)
    f.write_text(c, encoding='utf-8')
    print("patched OK")
else:
    print("pattern not found")
