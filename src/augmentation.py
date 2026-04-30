from pathlib import Path

import cv2
import numpy as np

from src.data_loader import gen_filepath_list
from src.utils import IMAGE_SIZE, image_extensions


def horizontal_flip(image):
    return np.fliplr(image)


def duplicate_flipped_images(image_dir, mask_dir=None, suffix="_hflip"):
    image_dir = Path(image_dir)
    created = []
    for image_path in gen_filepath_list(image_dir):
        if image_path.stem.endswith(suffix):
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        flipped = cv2.flip(image, 1)
        output_path = image_path.with_name(f"{image_path.stem}{suffix}{image_path.suffix}")
        cv2.imwrite(str(output_path), flipped)
        created.append(output_path)

    if mask_dir is not None:
        mask_dir = Path(mask_dir)
        for mask_path in gen_filepath_list(mask_dir):
            if mask_path.stem.endswith(suffix):
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            flipped_mask = cv2.flip(mask, 1)
            output_path = mask_path.with_name(f"{mask_path.stem}{suffix}{mask_path.suffix}")
            cv2.imwrite(str(output_path), flipped_mask)
            created.append(output_path)
    return created


def train_generator(
    image_dir,
    mask_dir,
    batch_size=8,
    seed=42,
    target_size=IMAGE_SIZE,
):
    """Generate images and masks together with the same seed and transform."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    image_datagen = ImageDataGenerator(rescale=1.0 / 255.0, horizontal_flip=True)
    mask_datagen = ImageDataGenerator(rescale=1.0 / 255.0, horizontal_flip=True)

    image_flow = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        color_mode="rgb",
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
    )
    mask_flow = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
    )

    for images, masks in zip(image_flow, mask_flow):
        masks = (masks > 0.5).astype("float32")
        yield images, masks
