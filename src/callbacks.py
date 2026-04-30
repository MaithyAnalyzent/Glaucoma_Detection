import numpy as np
from tensorflow.keras.callbacks import Callback


class BestH5ModelSaver(Callback):
    """Save a full model to .h5 when a monitored metric improves."""

    def __init__(self, filepath, monitor="val_loss", mode="min", verbose=1):
        super().__init__()
        self.filepath = str(filepath)
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = np.inf if mode == "min" else -np.inf

    def _is_improvement(self, current):
        if current is None:
            return False
        if self.mode == "min":
            return current < self.best
        return current > self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if self._is_improvement(current):
            previous = self.best
            self.best = current
            self.model.save(self.filepath)
            if self.verbose:
                print(
                    f"\nEpoch {epoch + 1}: {self.monitor} improved "
                    f"from {previous:.6f} to {current:.6f}; saved {self.filepath}"
                )
