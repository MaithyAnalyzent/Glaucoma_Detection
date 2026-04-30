import argparse
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import tensorflow as tf

from src.callbacks import BestH5ModelSaver
from src.data_loader import load_mask_pairs
from src.metrics import dice_score, iou, save_training_plots
from src.preprocess import read_image, read_mask
from src.unet_model import TL_unet_model, compile_unet
from src.utils import MASK_DIR, MODEL_DIR, PROCESSED_DIR, ensure_dirs, set_seed


def cpu_safe_batch_size(batch_size):
    if not tf.config.list_physical_devices("GPU") and batch_size > 1:
        print(
            f"No GPU detected. Reducing segmentation batch size from {batch_size} "
            "to 1 to avoid CPU memory exhaustion."
        )
        return 1
    return batch_size


class SegmentationSequence(Sequence):
    def __init__(self, pairs, batch_size=8, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.pairs = pairs.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch = self.pairs.iloc[batch_ids]
        images = np.array([read_image(path) for path in batch["image"]], dtype="float32")
        masks = np.array([read_mask(path) for path in batch["mask"]], dtype="float32")
        return images, masks

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def pretrain_unet(
    image_dir=PROCESSED_DIR,
    mask_dir=MASK_DIR,
    epochs=100,
    batch_size=8,
    learning_rate=1e-5,
    seed=42,
    max_samples=None,
    initial_epoch=0,
):
    ensure_dirs()
    set_seed(seed)
    batch_size = cpu_safe_batch_size(batch_size)
    pairs = load_mask_pairs(image_dir, mask_dir)
    if max_samples is not None:
        pairs = pairs.head(max_samples).reset_index(drop=True)
    train_pairs, test_pairs = train_test_split(
        pairs, test_size=0.2, random_state=seed, shuffle=True
    )

    train_seq = SegmentationSequence(train_pairs, batch_size=batch_size, shuffle=True)
    test_seq = SegmentationSequence(test_pairs, batch_size=batch_size, shuffle=False)

    checkpoint_path = MODEL_DIR / "unet_pretrained.h5"
    if initial_epoch > 0 and checkpoint_path.exists():
        print(f"Resuming from checkpoint {checkpoint_path} at epoch {initial_epoch}")
        model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={"dice_score": dice_score, "iou": iou},
        )
        compile_unet(model, learning_rate=learning_rate, metrics=[dice_score, iou])
    else:
        model = TL_unet_model((256, 256, 3))
        compile_unet(model, learning_rate=learning_rate, metrics=[dice_score, iou])
    checkpoint = BestH5ModelSaver(
        MODEL_DIR / "unet_pretrained.h5",
        monitor="val_loss",
        mode="min",
        verbose=1,
    )
    history = model.fit(
        train_seq,
        validation_data=test_seq,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint],
    )
    save_training_plots(history, "unet_pretrain")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Phase 1 U-Net pretraining")
    parser.add_argument("--image-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--mask-dir", default=str(MASK_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--initial-epoch", type=int, default=0)
    args = parser.parse_args()
    pretrain_unet(
        args.image_dir,
        args.mask_dir,
        args.epochs,
        args.batch_size,
        max_samples=args.max_samples,
        initial_epoch=args.initial_epoch,
    )


if __name__ == "__main__":
    main()
