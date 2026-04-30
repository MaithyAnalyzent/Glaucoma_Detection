import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence, to_categorical

from src.callbacks import BestH5ModelSaver
from src.cnn_model import build_cnn_classifier
from src.data_loader import load_brg_dataset
from src.metrics import save_training_plots, write_classification_report
from src.preprocess import read_image
from src.utils import CLASS_NAMES, MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR, ensure_dirs, set_seed


class ClassificationSequence(Sequence):
    def __init__(self, df, batch_size=16, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch = self.df.iloc[batch_ids]
        images = np.array([read_image(path) for path in batch["filepath"]], dtype="float32")
        labels = to_categorical(batch["label"].astype(int).values, num_classes=len(CLASS_NAMES))
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train_cnn(
    data_dir=PROCESSED_DIR,
    epochs=150,
    batch_size=16,
    seed=42,
    max_samples=None,
):
    ensure_dirs()
    set_seed(seed)
    df = load_brg_dataset(data_dir)
    if max_samples is not None:
        df = df.groupby("label", group_keys=False).head(max_samples // 2).reset_index(drop=True)
    train_val, test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=0.125, stratify=train_val["label"], random_state=seed
    )

    train_seq = ClassificationSequence(train, batch_size=batch_size, shuffle=True)
    val_seq = ClassificationSequence(val, batch_size=batch_size, shuffle=False)
    test_seq = ClassificationSequence(test, batch_size=batch_size, shuffle=False)

    model = build_cnn_classifier()
    checkpoint = BestH5ModelSaver(
        MODEL_DIR / "cnn_classifier.h5",
        monitor="val_accuracy",
        mode="max",
        verbose=1,
    )
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=[checkpoint],
    )
    save_training_plots(history, "cnn_classifier")

    y_true = test["label"].astype(int).values
    probabilities = model.predict(test_seq)
    y_pred = np.argmax(probabilities, axis=1)
    write_classification_report(
        y_true,
        y_pred,
        OUTPUT_DIR / "classification_report.txt",
    )
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train CNN glaucoma classifier")
    parser.add_argument("--data-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    train_cnn(Path(args.data_dir), args.epochs, args.batch_size, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
