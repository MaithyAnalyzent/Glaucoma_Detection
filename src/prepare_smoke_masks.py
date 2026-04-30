import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data_loader import gen_filepath_list
from src.utils import IMAGE_SIZE


def make_field_of_view_mask(image):
    """Create a simple binary field-of-view mask for code smoke testing only."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    return mask


def create_smoke_masks(image_dir, mask_dir, overwrite=False):
    """Create non-research masks so the segmentation code path can be tested.

    These masks are not clinical labels and must be replaced with real binary
    annotations for actual U-Net training.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    created = []

    for image_path in gen_filepath_list(image_dir):
        relative_path = image_path.relative_to(image_dir)
        mask_path = mask_dir / relative_path
        if mask_path.exists() and not overwrite:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        mask = make_field_of_view_mask(image)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask)
        created.append(mask_path)
    return created


def main():
    parser = argparse.ArgumentParser(
        description="Create smoke-test masks for checking the U-Net code path."
    )
    parser.add_argument("--image-dir", default="data/processed")
    parser.add_argument("--mask-dir", default="data/masks")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    created = create_smoke_masks(args.image_dir, args.mask_dir, args.overwrite)
    print(f"Smoke-test masks created: {len(created)}")
    print("Replace these with real binary segmentation masks for actual training.")


if __name__ == "__main__":
    main()
