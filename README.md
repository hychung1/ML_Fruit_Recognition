# Fruit and Vegetable Recognition (36 classes)

A university machine-learning project with **two independent approaches** to the same
Kaggle fruit/vegetable dataset, plus an Android app:

1. **Classical ML baselines** — handcrafted image features → KNN, SVM and Random Forest, with hyperparameter tuning and a side-by-side comparison.
2. **Deep learning** — EfficientNetB0 transfer learning → TFLite model → a Flutter app that classifies the **live camera feed on-device** (and a Streamlit web demo for single images).

## Repository layout

| Path | What it is |
| --- | --- |
| `dataset/` | Kaggle images (not in git — add locally, see below) |
| `eda.py` | Dataset statistics, class-distribution chart and sample grid → `results/` |
| `data_processing.py`, `data_visualization.py` | Original exploratory data generators / plots for the deep-learning path |
| `feature_extraction.py` | Extracts 800 features per image (512 HSV histogram + 288 HOG), caches them in `features/`, fits `StandardScaler` on train only → `models/scaler.joblib` |
| `train_knn.py` | Tunes k (1–21) and distance (Euclidean/Manhattan), evaluates on test → `models/knn_model.joblib` |
| `train_svm.py` | Tunes kernel/C/gamma, evaluates on test → `models/svm_model.joblib` |
| `train_rf.py` | Tunes trees/depth, reports feature importance, evaluates on test → `models/rf_model.joblib` |
| `compare_models.py` | Loads the three saved models and writes `results/metrics_summary.csv` + `results/comparison_chart.png` |
| `train_efficientnet.py` | Two-phase EfficientNetB0 training (frozen base, then fine-tune top layers) |
| `eval_efficientnet.py` | Test metrics + classification report + confusion matrix |
| `app.py` | Streamlit UI: upload an image, get EfficientNetB0 prediction |
| `convert_to_tflite.py` | Converts the Keras model to float16 TFLite, writes `fruit_recognition_app/assets/`, and checks Keras vs TFLite on one test image per class |
| `trained_model/` | Trained EfficientNetB0 checkpoints (`.h5`) |
| `results/`, `screenshots/` | Comparison tables, tuning plots, confusion matrices, training curves |
| `models/`, `features/` | Generated model/feature caches — ignored by git |
| `fruit_recognition_app/` | Flutter Android app (see its own [README](fruit_recognition_app/README.md)) |

## Results (classical models, test set)

From `results/metrics_summary.csv`:

| Model | Best parameters | Accuracy | Macro F1 | Speed |
| --- | --- | --- | --- | --- |
| KNN | k=1, Manhattan | 96.66% | 96.57% | 0.87 ms/image |
| SVM | linear, C=1.0 | 97.21% | 97.12% | 0.76 ms/image |
| Random Forest | 100 trees, max_depth=30 | 97.21% | 97.14% | 0.08 ms/image |

EfficientNetB0 metrics and curves are produced by `eval_efficientnet.py` and shown in
`screenshots/`. In the app, TFLite inference runs at roughly 8–10 fps on a phone.

## Setup

Use **Python 3.12**. TensorFlow 2.20 has no wheels for Python 3.13/3.14.

```powershell
python312 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Get the dataset from
[Kaggle — Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
and extract it so the layout is:

```text
dataset/
├── train/<36 class folders>/
├── validation/<36 class folders>/
└── test/<36 class folders>/
```

The dataset (~2 GB) is excluded from git via `.gitignore`.

## Workflow A — classical ML

Run from this folder, in order:

```powershell
.\.venv\Scripts\python eda.py
.\.venv\Scripts\python feature_extraction.py
.\.venv\Scripts\python train_knn.py
.\.venv\Scripts\python train_svm.py
.\.venv\Scripts\python train_rf.py
.\.venv\Scripts\python compare_models.py
```

`feature_extraction.py` caches features, so later runs are fast. Editing the dataset
requires `--force`-style re-extraction by calling `get_prepared_data(force_reload=True)`
or deleting `features/`.

Each `train_*.py` also exposes `predict_image_*()` helpers (used to integrate a chosen
model into a web app). `app.py` currently demonstrates the EfficientNetB0 model only.

## Workflow B — EfficientNetB0 + Android app

```powershell
.\.venv\Scripts\python train_efficientnet.py       # optional: retrain the .h5
.\.venv\Scripts\python eval_efficientnet.py
.\.venv\Scripts\python convert_to_tflite.py        # refresh assets/fruit_model.tflite + labels.txt
.\.venv\Scripts\python -m streamlit run app.py     # web demo
```

Then build the app (requires Flutter SDK and an Android device):

```powershell
cd fruit_recognition_app
flutter pub get
flutter run
flutter build apk --release --split-per-abi
```

The APK is **not** stored in git; build it or download the published arm64 APK from this
repository's GitHub Releases.

Note: in TensorFlow/Keras 3, `efficientnet.preprocess_input` is a no-op because the
rescaling/normalization is inside the model. Both `convert_to_tflite.py` and the Flutter
app feed raw 0–255 RGB pixels — do not apply mean/std normalization a second time.

## Related reference projects

- CNN: https://github.com/ondrejhruby/Fruit-Recognition.git
- MobileNetV2: https://github.com/mahmouddbelo/Fruit-Classification-Using-MobileNetV2-.git
- EfficientNetB0: https://github.com/seanpan-1321/fruit-classifier-efficientnetb0.git
