# Retrained U-Net Neural Network to Detect Glaucoma Retinal Disorder with OCTA Eye Images

This project recreates the methodology described in the academic report:

```text
Eye Image
-> Image Preprocessing
-> Image Segmentation using V-UNET V1.1
-> Preprocessed image data samples
-> Train / Validation / Test split
-> CNN classifier
-> Normal / Glaucoma prediction
```

No transformers, GradCAM, SHAP, explainable AI, ensembles, extra datasets, or modern replacement architectures are included.

## Important Dataset Note

The report names the Brazil Glaucoma Database (BrG). The public BrG dataset available from Kaggle is fundus retinal imagery, not OCT/OCTA volume data.

The downloaded BrG dataset contains class images but does not include ground-truth segmentation masks. Full U-Net training requires real binary masks in `data/masks/` with filenames matching the source image stems. Do not generate fake masks for real training.

## Project Structure

```text
glaucoma_project/
  app.py
  requirements.txt
  README.md
  data/
    raw/
    processed/
    masks/
  models/
  notebooks/
  outputs/
    plots/
    segmented/
  src/
    augmentation.py
    cnn_model.py
    data_loader.py
    download_dataset.py
    finetune.py
    metrics.py
    predict.py
    preprocess.py
    train_cnn.py
    train_unet.py
    unet_model.py
    utils.py
```

## 1. Open The Project Folder

From PowerShell:

```powershell
cd "E:\Perplexity Projects\Glaucoma_Detection\glaucoma_project"
```

## 2. Install Python Dependencies

Use Python 3.12 on this machine. TensorFlow was tested with Python 3.12 here.

```powershell
py -3.12 -m pip install -r requirements.txt --user
py -3.12 -m pip install kaggle --user
```

Quick dependency check:

```powershell
py -3.12 -c "import tensorflow as tf, cv2, streamlit; print(tf.__version__)"
```

## 3. Download BrG Dataset

Dataset source:

- Kaggle dataset: `clerimar/brasil-glaucoma-brg`
- Title: Brasil Glaucoma BrG
- License shown by Kaggle: CC BY-NC 4.0
- Contents: 2,000 retinal fundus images from 1,000 volunteers

If Kaggle asks for credentials, create a Kaggle API token from your Kaggle account and place `kaggle.json` in:

```text
C:\Users\<your-user-name>\.kaggle\kaggle.json
```

Download and unzip:

```powershell
& "$env:APPDATA\Python\Python312\Scripts\kaggle.exe" datasets download -d clerimar/brasil-glaucoma-brg -p data --unzip
```

If the dataset extracts as `data/BrG_Dataset`, move it into `data/raw`:

```powershell
Move-Item -LiteralPath data\BrG_Dataset -Destination data\raw\BrG_Dataset
```

You can also use the project downloader:

```powershell
py -3.12 src/download_dataset.py --destination data/raw
```

Verify the raw dataset:

```powershell
py -3.12 -c "from src.download_dataset import summarize_dataset; print(summarize_dataset('data/raw'))"
```

Expected raw count:

```text
2000 images
1000 glaucoma
1000 normal
```

## 4. Preprocess Images

This resizes images into `data/processed`. Pixel normalization is performed by the model input loaders during training and prediction.

```powershell
py -3.12 -c "from src.preprocess import preprocess_directory; saved=preprocess_directory('data/raw','data/processed'); print(len(saved))"
```

Expected output before augmentation:

```text
2000
```

## 5. Apply Horizontal Flip Augmentation

This duplicates image copies using horizontal flip.

```powershell
py -3.12 -c "from src.augmentation import duplicate_flipped_images; created=duplicate_flipped_images('data/processed'); print(len(created))"
```

Expected output:

```text
2000
```

Verify processed data:

```powershell
py -3.12 -c "from src.data_loader import load_brg_dataset; df=load_brg_dataset('data/processed'); print(df['label'].value_counts().to_dict()); print(len(df))"
```

Expected processed count:

```text
{1: 2000, 0: 2000}
4000
```

## 6. Add Segmentation Masks For Full U-Net Training

To run the full report pipeline, place real binary masks in:

```text
data/masks/
```

Mask filenames must match image stems. Example:

```text
data/processed/.../imGOE (1).png
data/masks/.../imGOE (1).png
```

For image and mask augmentation together:

```powershell
py -3.12 -c "from src.augmentation import duplicate_flipped_images; duplicate_flipped_images('data/processed','data/masks')"
```

For a code-only smoke test, create simple field-of-view masks:

```powershell
py -3.12 -m src.prepare_smoke_masks --image-dir data/processed --mask-dir data/masks
```

These smoke-test masks are not research labels. Replace them with real binary
segmentation masks before reporting U-Net training results.

## 7. Smoke Test Model Construction

Run this before long training:

```powershell
$env:TF_CPP_MIN_LOG_LEVEL='2'
py -3.12 -c "from src.unet_model import TL_unet_model; from src.cnn_model import build_cnn_classifier; import numpy as np; u=TL_unet_model((256,256,3)); c=build_cnn_classifier(); y=u.predict(np.zeros((1,256,256,3), dtype='float32'), verbose=0); print(u.output_shape); print(c.output_shape); print(y.shape)"
```

Expected shapes:

```text
(None, 256, 256, 1)
(None, 2)
(1, 256, 256, 1)
```

## 8. Train U-Net Phase 1

Requires real masks in `data/masks`.

Configuration from the report:

- Model: V-UNET V1.1
- Encoder: pretrained VGG16
- Encoder state: frozen initially
- Train-test split: 0.8
- Learning rate: `1e-5`
- Epochs: `100`

Run:

```powershell
py -3.12 -m src.train_unet --image-dir data/processed --mask-dir data/masks --epochs 20 --batch-size 8
```

On CPU-only machines, the script automatically reduces segmentation batch size
to `1` to avoid TensorFlow memory exhaustion.

Saves:

```text
models/unet_pretrained.h5
outputs/plots/unet_pretrain_loss.png
```

## 9. Fine Tune U-Net Phase 2

Requires real masks in `data/masks`.

Configuration from the report:

- Unfreeze contracting path
- Function: `finetune_unfreezeall()`
- KFold cross validation: `k=5`
- Learning rate: `5e-6`
- Epochs: `100`
- Metrics: dice score and IoU

Run:

```powershell
py -3.12 -m src.finetune --image-dir data/processed --mask-dir data/masks --epochs 100 --batch-size 8
```

On CPU-only machines, the script automatically reduces segmentation batch size
to `1` to avoid TensorFlow memory exhaustion.

Saves:

```text
models/unet_finetuned.h5
models/unet_finetuned_fold_*.h5
outputs/plots/fold_training_results.png
```

## 10. Generate Segmented Images

Requires `models/unet_finetuned.h5`.

```powershell
py -3.12 -c "from src.predict import load_unet, generate_segmented_dataset; generate_segmented_dataset('data/processed', load_unet(), 'outputs/segmented')"
```

Saves segmented classifier inputs to:

```text
outputs/segmented/
```

## 11. Train CNN Classifier

Configuration from the report:

- Convolutional layers
- Pooling layers
- Fully connected layers
- ReLU activation
- Final Softmax layer
- Optimizer: Adam
- Epochs: `150`

Preferred full-pipeline input is segmented data:

```powershell
py -3.12 -m src.train_cnn --data-dir outputs/segmented --epochs 150 --batch-size 16
```

If segmentation masks/models are not available yet, you can verify the CNN training path directly on processed BrG images:

```powershell
py -3.12 -m src.train_cnn --data-dir data/processed --epochs 150 --batch-size 16
```

Saves:

```text
models/cnn_classifier.h5
outputs/classification_report.txt
outputs/plots/cnn_classifier_accuracy.png
outputs/plots/cnn_classifier_loss.png
```

## 12. Quick CNN Smoke Test

This runs one training step on a tiny real-image sample.

```powershell
$env:TF_CPP_MIN_LOG_LEVEL='2'
py -3.12 -c "import pandas as pd; from src.data_loader import load_brg_dataset; from src.train_cnn import ClassificationSequence; from src.cnn_model import build_cnn_classifier; df=load_brg_dataset('data/processed'); sample=pd.concat([df[df.label==0].head(8), df[df.label==1].head(8)]).reset_index(drop=True); seq=ClassificationSequence(sample, batch_size=4, shuffle=False); model=build_cnn_classifier(); h=model.fit(seq, epochs=1, steps_per_epoch=1, verbose=0); print(float(h.history['loss'][0])); print(model.predict(seq, steps=1, verbose=0).shape)"
```

Expected final probability shape:

```text
(4, 2)
```

## 13. Run Prediction From Command Line

Requires:

```text
models/unet_finetuned.h5
models/cnn_classifier.h5
```

Run:

```powershell
py -3.12 -m src.predict "path\to\eye_image.png"
```

Output:

```text
Prediction: Normal or Glaucoma
Confidence: score
```

## 14. Run Streamlit Frontend

Use the same Python version used for installation:

```powershell
py -3.12 -m streamlit run app.py
```

Pages:

- Home
- Upload Eye Image
- Prediction
- Metrics

The prediction page requires both trained models:

```text
models/unet_finetuned.h5
models/cnn_classifier.h5
```

## 15. Notebook Workflow

Run notebooks in this order:

```text
notebooks/01_dataset.ipynb
notebooks/02_unet_training.ipynb
notebooks/03_finetune_kfold.ipynb
notebooks/04_cnn_training.ipynb
notebooks/05_results.ipynb
```

## 16. Outputs

After a complete run:

```text
models/unet_pretrained.h5
models/unet_finetuned.h5
models/cnn_classifier.h5
outputs/segmented/
outputs/plots/
outputs/classification_report.txt
```

The report states a target reference accuracy of `98.47%`. Actual accuracy depends on the available masks, exact data split, hardware, training completion, and random seed.
