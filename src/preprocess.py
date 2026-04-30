from pathlib import Path

import cv2
import numpy as np

from src.utils import IMAGE_SIZE, PROCESSED_DIR, ensure_dirs, image_extensions


def resize_image(image, size=IMAGE_SIZE):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image):
    return image.astype("float32") / 255.0


def read_image(path, size=IMAGE_SIZE, grayscale=False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    image = resize_image(image, size)
    if not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return normalize_image(image)


def read_mask(path, size=IMAGE_SIZE):
    mask = read_image(path, size=size, grayscale=True)
    mask = (mask > 0.5).astype("float32")
    return np.expand_dims(mask, axis=-1)


def preprocess_directory(input_dir, output_dir=PROCESSED_DIR, size=IMAGE_SIZE):
    ensure_dirs()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in image_extensions():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = resize_image(image, size)
        target = output_dir / path.relative_to(input_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), image)
        saved.append(target)
    return saved
