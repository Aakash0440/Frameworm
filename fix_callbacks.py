import pathlib

f = pathlib.Path("training/callbacks.py")
c = f.read_text(encoding="utf-8")
early_stop = '''

class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving."""
    def __init__(self, patience=5, min_delta=0.0, monitor="val_loss", mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def on_epoch_end(self, epoch, trainer):
        metrics = getattr(trainer.state, "val_metrics", {})
        val = metrics.get(self.monitor) or metrics.get("loss")
        if val is None:
            return
        val = val[-1] if isinstance(val, list) else val
        improved = val < self.best - self.min_delta if self.mode == "min" else val > self.best + self.min_delta
        if improved:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer._stop_training = True
'''
if "class EarlyStopping" not in c:
    c += early_stop
    f.write_text(c, encoding="utf-8")
    print("EarlyStopping added")
else:
    print("already exists")
