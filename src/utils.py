import os
import random
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MASK_DIR = DATA_DIR / "masks"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SEGMENTED_DIR = OUTPUT_DIR / "segmented"
PLOTS_DIR = OUTPUT_DIR / "plots"

IMAGE_SIZE = (256, 256)
CLASS_NAMES = ["Normal", "Glaucoma"]


def ensure_dirs():
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        MASK_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        SEGMENTED_DIR,
        PLOTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def image_extensions():
    return {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def label_from_path(path):
    lower = str(path).lower()
    if "glaucoma" in lower and "non" not in lower and "normal" not in lower:
        return 1
    if "normal" in lower or "non-glaucoma" in lower or "nonglaucoma" in lower:
        return 0
    raise ValueError(
        f"Cannot infer label for {path}. Place files inside Normal/ and Glaucoma/ folders."
    )
