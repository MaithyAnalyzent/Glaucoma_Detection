# Python Commands To Run The Project

Run all commands from the project folder:

```powershell
cd "E:\Perplexity Projects\Glaucoma_Detection\glaucoma_project"
```

## 1. Install Requirements

```powershell
py -3.12 -m pip install -r requirements.txt --user
py -3.12 -m pip install kaggle --user
```

## 2. Check Dependencies

```powershell
py -3.12 -c "import tensorflow as tf, cv2, streamlit; print(tf.__version__)"
```

## 3. Download BrG Dataset

```powershell
& "$env:APPDATA\Python\Python312\Scripts\kaggle.exe" datasets download -d clerimar/brasil-glaucoma-brg -p data --unzip
```

If needed, move extracted data into `data/raw`:

```powershell
Move-Item -LiteralPath data\BrG_Dataset -Destination data\raw\BrG_Dataset
```

Alternative project downloader:

```powershell
py -3.12 src/download_dataset.py --destination data/raw
```

## 4. Verify Raw Dataset

```powershell
py -3.12 -c "from src.download_dataset import summarize_dataset; print(summarize_dataset('data/raw'))"
```

## 5. Preprocess Images

```powershell
py -3.12 -c "from src.preprocess import preprocess_directory; saved=preprocess_directory('data/raw','data/processed'); print(len(saved))"
```

## 6. Horizontal Flip Augmentation

```powershell
py -3.12 -c "from src.augmentation import duplicate_flipped_images; created=duplicate_flipped_images('data/processed'); print(len(created))"
```

## 7. Verify Processed Dataset

```powershell
py -3.12 -c "from src.data_loader import load_brg_dataset; df=load_brg_dataset('data/processed'); print(df['label'].value_counts().to_dict()); print(len(df))"
```

## 8. Smoke Test Models

```powershell
$env:TF_CPP_MIN_LOG_LEVEL='2'
py -3.12 -c "from src.unet_model import TL_unet_model; from src.cnn_model import build_cnn_classifier; import numpy as np; u=TL_unet_model((256,256,3)); c=build_cnn_classifier(); y=u.predict(np.zeros((1,256,256,3), dtype='float32'), verbose=0); print(u.output_shape); print(c.output_shape); print(y.shape)"
```

## 9. Create Smoke-Test Masks

Use this only to test code execution when real masks are unavailable.

```powershell
py -3.12 -m src.prepare_smoke_masks --image-dir data/processed --mask-dir data/masks
```

## 10. Train U-Net Phase 1

Requires real masks in `data/masks`.

```powershell
py -3.12 -m src.train_unet --image-dir data/processed --mask-dir data/masks --epochs 100 --batch-size 8
```

On CPU-only machines, the script automatically reduces segmentation batch size to `1`.

Quick smoke test:

```powershell
py -3.12 -m src.train_unet --image-dir data/processed --mask-dir data/masks --epochs 1 --batch-size 8 --max-samples 5
```

## 11. Fine Tune U-Net Phase 2

Requires real masks in `data/masks`.

```powershell
py -3.12 -m src.finetune --image-dir data/processed --mask-dir data/masks --epochs 100 --batch-size 8
```

On CPU-only machines, the script automatically reduces segmentation batch size to `1`.

## 12. Generate Segmented Images

Requires `models/unet_finetuned.h5`.

```powershell
py -3.12 -c "from src.predict import load_unet, generate_segmented_dataset; generate_segmented_dataset('data/processed', load_unet(), 'outputs/segmented')"
```

## 13. Train CNN Classifier

Full report pipeline input:

```powershell
py -3.12 -m src.train_cnn --data-dir outputs/segmented --epochs 150 --batch-size 16
```

CNN-only test input using processed BrG images:

```powershell
py -3.12 -m src.train_cnn --data-dir data/processed --epochs 150 --batch-size 16
```

## 14. Quick CNN Smoke Test

```powershell
$env:TF_CPP_MIN_LOG_LEVEL='2'
py -3.12 -c "import pandas as pd; from src.data_loader import load_brg_dataset; from src.train_cnn import ClassificationSequence; from src.cnn_model import build_cnn_classifier; df=load_brg_dataset('data/processed'); sample=pd.concat([df[df.label==0].head(8), df[df.label==1].head(8)]).reset_index(drop=True); seq=ClassificationSequence(sample, batch_size=4, shuffle=False); model=build_cnn_classifier(); h=model.fit(seq, epochs=1, steps_per_epoch=1, verbose=0); print(float(h.history['loss'][0])); print(model.predict(seq, steps=1, verbose=0).shape)"
```

## 15. Predict One Image

Requires:

```text
models/unet_finetuned.h5
models/cnn_classifier.h5
```

```powershell
py -3.12 -m src.predict "path\to\eye_image.png"
```

## 16. Run Streamlit App

```powershell
py -3.12 -m streamlit run app.py
```
