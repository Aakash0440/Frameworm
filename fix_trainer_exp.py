import pathlib

f = pathlib.Path("training/trainer.py")
c = f.read_text(encoding="utf-8")
old = "        self.train_tracker.update(loss_dict)"
new = "        self.train_tracker.update(loss_dict)\n            if self.experiment:\n                for k, v in loss_dict.items():\n                    if hasattr(v, 'numel') and v.numel() != 1:\n                        continue\n                    self.experiment.log_metric(k, float(v), step=self.state.global_step, metric_type='train')"
if old in c:
    c = c.replace(old, new)
    f.write_text(c, encoding="utf-8")
    print("experiment logging added")
else:
    print("pattern not found")
