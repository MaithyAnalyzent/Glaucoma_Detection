import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src.utils import RAW_DIR, ensure_dirs, image_extensions


BRG_KAGGLE_SLUG = "clerimar/brasil-glaucoma-brg"


def _run(command):
    return subprocess.run(command, check=True, text=True)


def install_kaggle_if_missing():
    if find_kaggle_executable() is not None:
        return
    _run([sys.executable, "-m", "pip", "install", "kaggle"])


def find_kaggle_executable():
    executable = shutil.which("kaggle")
    if executable is not None:
        return executable
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidate = Path(appdata) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "kaggle.exe"
        if candidate.exists():
            return str(candidate)
    return None


def download_brg_dataset(destination=RAW_DIR):
    """Download the BrG dataset from Kaggle and extract it into data/raw.

    Kaggle requires API credentials in ~/.kaggle/kaggle.json or equivalent
    environment variables before this command can download the dataset.
    """
    ensure_dirs()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    install_kaggle_if_missing()

    zip_path = destination / "brasil-glaucoma-brg.zip"
    kaggle_executable = find_kaggle_executable()
    if kaggle_executable is None:
        raise RuntimeError("Kaggle executable was not found after installation.")
    _run(
        [
            kaggle_executable,
            "datasets",
            "download",
            "-d",
            BRG_KAGGLE_SLUG,
            "-p",
            str(destination),
            "--force",
        ]
    )
    downloaded = next(destination.glob("*.zip"), None)
    if downloaded is None:
        raise FileNotFoundError("Kaggle did not create a zip file.")
    if downloaded != zip_path:
        downloaded.rename(zip_path)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)
    return destination


def summarize_dataset(data_dir=RAW_DIR):
    data_dir = Path(data_dir)
    image_count = 0
    by_folder = {}
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_extensions():
            image_count += 1
            key = str(path.parent.relative_to(data_dir))
            by_folder[key] = by_folder.get(key, 0) + 1
    return {"image_count": image_count, "by_folder": by_folder}


def main():
    parser = argparse.ArgumentParser(description="Download and summarize BrG dataset")
    parser.add_argument("--destination", default=str(RAW_DIR))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if not args.summary_only:
        download_brg_dataset(args.destination)
    summary = summarize_dataset(args.destination)
    print(f"Images found: {summary['image_count']}")
    for folder, count in sorted(summary["by_folder"].items()):
        print(f"{folder}: {count}")


if __name__ == "__main__":
    main()
