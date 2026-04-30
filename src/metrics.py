import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

from src.utils import CLASS_NAMES, PLOTS_DIR, ensure_dirs


def dice_score(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (denominator + smooth)


def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def save_training_plots(history, prefix):
    ensure_dirs()
    history_dict = history.history if hasattr(history, "history") else history

    if "accuracy" in history_dict:
        plt.figure()
        plt.plot(history_dict["accuracy"], label="train")
        if "val_accuracy" in history_dict:
            plt.plot(history_dict["val_accuracy"], label="validation")
        plt.title("Model accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{prefix}_accuracy.png")
        plt.close()

    plt.figure()
    plt.plot(history_dict["loss"], label="train")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="validation")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{prefix}_loss.png")
    plt.close()


def save_fold_results(fold_scores, output_path=None):
    ensure_dirs()
    output_path = output_path or PLOTS_DIR / "fold_training_results.png"
    folds = np.arange(1, len(fold_scores) + 1)
    dice_values = [score.get("dice_score", 0.0) for score in fold_scores]
    iou_values = [score.get("iou", 0.0) for score in fold_scores]

    plt.figure()
    plt.plot(folds, dice_values, marker="o", label="Dice score")
    plt.plot(folds, iou_values, marker="o", label="IoU")
    plt.title("Fold training results")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.xticks(folds)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_classification_report(y_true, y_pred, output_path):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    matrix = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Classification Report\n")
        file.write("=====================\n\n")
        file.write(report)
        file.write("\nConfusion Matrix\n")
        file.write(str(matrix))
        file.write("\n\nSummary Metrics\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value:.6f}\n")
        file.write("\nTarget reference accuracy stated in report: 98.47%\n")
    return metrics, matrix
