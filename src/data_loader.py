from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import RAW_DIR, image_extensions, label_from_path


def gen_filepath_list(data_dir=RAW_DIR):
    """Return image file paths for BrG-style folders."""
    data_dir = Path(data_dir)
    files = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions()
    ]
    return sorted(files)


def load_brg_dataset(data_dir=RAW_DIR):
    """Load Brazil Glaucoma Database paths and labels.

    Expected layout:
        data/raw/Normal/*.png
        data/raw/Glaucoma/*.png

    Left and right eye images can be kept in the same class folders or in
    subfolders below the class name.
    """
    rows = []
    for path in gen_filepath_list(data_dir):
        rows.append({"filepath": str(path), "label": label_from_path(path)})
    if not rows:
        raise FileNotFoundError(
            f"No image files found under {data_dir}. Add BrG images before training."
        )
    return pd.DataFrame(rows)


def split_dataset(df, test_size=0.2, val_size=0.1, seed=42):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_mask_pairs(image_dir, mask_dir):
    """Pair each image with a mask.

    Preferred pairing is by relative path, which is important for BrG because
    left/right and class folders can contain repeated filename stems. If masks
    are stored in a flat folder, unique filename stems are used as a fallback.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_paths = gen_filepath_list(image_dir)
    mask_files = gen_filepath_list(mask_dir)
    mask_by_relative = {}
    for mask_path in mask_files:
        try:
            mask_by_relative[mask_path.relative_to(mask_dir).as_posix().lower()] = mask_path
        except ValueError:
            continue

    masks_by_stem = {}
    duplicate_stems = set()
    for mask_path in mask_files:
        key = mask_path.stem.lower()
        if key in masks_by_stem:
            duplicate_stems.add(key)
        masks_by_stem[key] = mask_path

    pairs = []
    for image_path in image_paths:
        relative_key = image_path.relative_to(image_dir).as_posix().lower()
        mask_path = mask_by_relative.get(relative_key)
        if mask_path is None and image_path.stem.lower() not in duplicate_stems:
            mask_path = masks_by_stem.get(image_path.stem.lower())
        if mask_path is not None:
            pairs.append((str(image_path), str(mask_path)))
    if not pairs:
        raise FileNotFoundError(
            "No image/mask pairs found.\n"
            f"Images found under {image_dir}: {len(image_paths)}\n"
            f"Masks found under {mask_dir}: {len(mask_files)}\n"
            "For full U-Net training, add real binary masks in data/masks with "
            "the same relative paths or unique filename stems as data/processed.\n"
            "For a code-only smoke test, run:\n"
            "py -3.12 -m src.prepare_smoke_masks --image-dir data/processed "
            "--mask-dir data/masks"
        )
    return pd.DataFrame(pairs, columns=["image", "mask"])
