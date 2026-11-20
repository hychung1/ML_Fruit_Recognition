# Fruit and Vegetable Recognition (36 classes)

A university machine-learning project with **two independent approaches** to the same
Kaggle fruit/vegetable dataset, plus an Android app:

1. **Classical ML baselines** — handcrafted image features → KNN, SVM and Random Forest, with hyperparameter tuning and a side-by-side comparison.
2. **Deep learning** — EfficientNetB0 transfer learning → TFLite model → a Flutter app that classifies the **live camera feed on-device** (and a Streamlit web demo for single images).

## Repository layout

All Python code lives in two script folders; data and generated files stay at the root so
both pipelines share the same dataset.

| Path | What it is |
| --- | --- |
| `classical_ml/` | KNN / SVM / Random Forest pipeline (see files below) |
| `deep_learning/` | EfficientNetB0 training, evaluation, export and Streamlit demo |
| `dataset/` | Kaggle images (not in git — add locally, see below) |
| `results/` | Classical-model metrics and plots |
| `results_dl/` | EfficientNetB0 training curve and confusion matrix |
| `dl_model/` | Trained EfficientNetB0 checkpoints (`.h5`) |
| `models/` | Trained KNN, SVM and Random Forest models + `scaler.joblib` (committed, ready to use) |
| `features/` | Generated feature caches — ignored by git |
| `fruit_recognition_app/` | Flutter Android app (see its own [README](fruit_recognition_app/README.md)) |

`classical_ml/`:

| Script | What it does |
| --- | --- |
| `eda.py` | Dataset statistics, class-distribution chart and sample grid → `results/` |
| `feature_extraction.py` | 800 features per image (512 HSV histogram + 288 HOG), cached in `features/`; fits `StandardScaler` on train only → `models/scaler.joblib` |
| `train_knn.py` | Tunes k (1–21) and distance, evaluates on test → `models/knn_model.joblib` |
| `train_svm.py` | Tunes kernel/C/gamma, evaluates on test → `models/svm_model.joblib` |
| `train_rf.py` | Tunes trees/depth, reports feature importance → `models/rf_model.joblib` |
| `compare_models.py` | Writes `results/metrics_summary.csv` + `results/comparison_chart.png` |

`deep_learning/`:

| Script | What it does |
| --- | --- |
| `data_processing.py`, `data_visualization.py` | Image generators (augmentation) and exploratory plots |
| `train_efficientnet.py` | Two-phase EfficientNetB0 training → `dl_model/` |
| `eval_efficientnet.py` | Test metrics, classification report, confusion matrix → `results_dl/` |
| `convert_to_tflite.py` | Float16 TFLite export → `fruit_recognition_app/assets/`, with a Keras-vs-TFLite check |
| `app.py` | Streamlit UI: upload an image, get EfficientNetB0 prediction |

## Results (classical models, test set)

From `results/metrics_summary.csv`:

| Model | Best parameters | Accuracy | Macro F1 | Speed |
| --- | --- | --- | --- | --- |
| KNN | k=1, Manhattan | 96.66% | 96.57% | 0.87 ms/image |
| SVM | linear, C=1.0 | 97.21% | 97.12% | 0.76 ms/image |
| Random Forest | 100 trees, max_depth=30 | 97.21% | 97.14% | 0.08 ms/image |

EfficientNetB0 metrics and curves are produced by `eval_efficientnet.py` and shown in
`results_dl/`. In the app, TFLite inference runs at roughly 8–10 fps on a phone.

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
.\.venv\Scripts\python classical_ml\eda.py
.\.venv\Scripts\python classical_ml\feature_extraction.py
.\.venv\Scripts\python classical_ml\train_knn.py
.\.venv\Scripts\python classical_ml\train_svm.py
.\.venv\Scripts\python classical_ml\train_rf.py
.\.venv\Scripts\python classical_ml\compare_models.py
```

All paths resolve relative to this folder, no matter where you launch Python from.
`feature_extraction.py` caches features, so later runs are fast. After editing the
dataset, delete `features/` to force re-extraction.

Each `train_*.py` also exposes `predict_image_*()` helpers (used to integrate a chosen
model into a web app). `deep_learning/app.py` currently demonstrates EfficientNetB0 only.

## Workflow B — EfficientNetB0 + Android app

```powershell
.\.venv\Scripts\python deep_learning\train_efficientnet.py  # optional: retrain the .h5
.\.venv\Scripts\python deep_learning\eval_efficientnet.py
.\.venv\Scripts\python deep_learning\convert_to_tflite.py   # refresh assets/fruit_model.tflite + labels.txt
.\.venv\Scripts\python -m streamlit run deep_learning\app.py # web demo
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
