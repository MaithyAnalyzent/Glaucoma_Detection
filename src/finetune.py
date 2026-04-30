import argparse
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

from src.callbacks import BestH5ModelSaver
from src.data_loader import load_mask_pairs
from src.metrics import dice_score, iou, save_fold_results, save_training_plots
from src.train_unet import SegmentationSequence
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


def finetune_unfreezeall(model):
    """Unfreeze all layers of the VGG16 contracting path/base model."""
    for layer in model.layers:
        layer.trainable = True
    return model


def finetune_kfold(
    image_dir=PROCESSED_DIR,
    mask_dir=MASK_DIR,
    epochs=100,
    batch_size=8,
    learning_rate=5e-6,
    k=5,
    seed=42,
    max_samples=None,
):
    ensure_dirs()
    set_seed(seed)
    batch_size = cpu_safe_batch_size(batch_size)
    pairs = load_mask_pairs(image_dir, mask_dir)
    if max_samples is not None:
        pairs = pairs.head(max_samples).reset_index(drop=True)
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_scores = []
    best_val_loss = np.inf

    for fold, (train_idx, val_idx) in enumerate(kfold.split(pairs), start=1):
        train_pairs = pairs.iloc[train_idx].reset_index(drop=True)
        val_pairs = pairs.iloc[val_idx].reset_index(drop=True)
        train_seq = SegmentationSequence(train_pairs, batch_size=batch_size, shuffle=True)
        val_seq = SegmentationSequence(val_pairs, batch_size=batch_size, shuffle=False)

        model = TL_unet_model((256, 256, 3))
        finetune_unfreezeall(model)
        compile_unet(model, learning_rate=learning_rate, metrics=[dice_score, iou])

        checkpoint_path = MODEL_DIR / f"unet_finetuned_fold_{fold}.h5"
        checkpoint = BestH5ModelSaver(
            checkpoint_path,
            monitor="val_loss",
            mode="min",
            verbose=1,
        )
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=epochs,
            callbacks=[checkpoint],
        )
        save_training_plots(history, f"unet_finetune_fold_{fold}")
        results = model.evaluate(val_seq, verbose=0)
        score_map = dict(zip(model.metrics_names, results))
        fold_scores.append(score_map)

        val_loss = score_map.get("loss", np.inf)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(MODEL_DIR / "unet_finetuned.h5")

    save_fold_results(fold_scores)
    return fold_scores


def main():
    parser = argparse.ArgumentParser(description="Phase 2 U-Net KFold fine tuning")
    parser.add_argument("--image-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--mask-dir", default=str(MASK_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    finetune_kfold(
        args.image_dir,
        args.mask_dir,
        args.epochs,
        args.batch_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
