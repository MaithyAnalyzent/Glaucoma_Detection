import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.metrics import dice_score, iou
from src.preprocess import read_image
from src.utils import CLASS_NAMES, MODEL_DIR, SEGMENTED_DIR, ensure_dirs, image_extensions


def load_unet(path=MODEL_DIR / "unet_finetuned.h5"):
    return load_model(path, custom_objects={"dice_score": dice_score, "iou": iou})


def load_classifier(path=MODEL_DIR / "cnn_classifier.h5"):
    return load_model(path)


def segment_image(image_path, unet_model, threshold=0.5):
    image = read_image(image_path)
    prediction = unet_model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    binary_mask = (prediction > threshold).astype("float32")
    segmented = image * binary_mask
    return image, binary_mask, segmented


def save_segmented_sample(image_path, unet_model, output_dir=SEGMENTED_DIR):
    ensure_dirs()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image, mask, segmented = segment_image(image_path, unet_model)
    stem = Path(image_path).stem

    original_out = output_dir / f"{stem}_original.png"
    mask_out = output_dir / f"{stem}_mask.png"
    segmented_out = output_dir / f"{stem}_segmented.png"

    cv2.imwrite(str(original_out), cv2.cvtColor((image * 255).astype("uint8"), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_out), (mask[:, :, 0] * 255).astype("uint8"))
    cv2.imwrite(
        str(segmented_out),
        cv2.cvtColor((segmented * 255).astype("uint8"), cv2.COLOR_RGB2BGR),
    )
    return original_out, mask_out, segmented_out


def generate_segmented_dataset(image_dir, unet_model, output_dir=SEGMENTED_DIR):
    image_dir = Path(image_dir)
    saved = []
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in image_extensions():
            continue
        class_dir = path.parent.name
        target_dir = Path(output_dir) / class_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        _, _, segmented = segment_image(path, unet_model)
        output_path = target_dir / f"{path.stem}_segmented.png"
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor((segmented * 255).astype("uint8"), cv2.COLOR_RGB2BGR),
        )
        saved.append(output_path)
    return saved


def predict_image(image_path, unet_model=None, classifier_model=None):
    if unet_model is None:
        unet_model = load_unet()
    if classifier_model is None:
        classifier_model = load_classifier()

    image, mask, segmented = segment_image(image_path, unet_model)
    probabilities = classifier_model.predict(np.expand_dims(segmented, axis=0), verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return {
        "image": image,
        "mask": mask,
        "segmented": segmented,
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": probabilities,
    }


def main():
    parser = argparse.ArgumentParser(description="Segment and classify one retinal image")
    parser.add_argument("image_path")
    parser.add_argument("--unet-model", default=str(MODEL_DIR / "unet_finetuned.h5"))
    parser.add_argument("--cnn-model", default=str(MODEL_DIR / "cnn_classifier.h5"))
    args = parser.parse_args()
    result = predict_image(
        args.image_path,
        load_unet(args.unet_model),
        load_classifier(args.cnn_model),
    )
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
