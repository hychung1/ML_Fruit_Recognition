# fruit_recognition_app

Real-time fruit and vegetable scanner for Android (**v1.1.0**). The camera feed
is classified on-device with the EfficientNetB0 model trained in the parent
folder, converted to TFLite. v1.1.0 adds multi-fruit detection: up to three
fruits in one camera view each get a green box and a label with confidence, and
the launcher icon uses the new artwork in `assets/app_icon.png`.

## Run

```sh
flutter pub get
flutter run
```

The app needs a physical Android device (back camera). It is locked to portrait.

## Build an APK

```sh
# one APK per CPU architecture (smallest; use this for your own phone)
flutter build apk --release --split-per-abi

# single APK that works on any phone (larger)
flutter build apk --release
```

Output goes to `build/app/outputs/flutter-apk/`:

| APK | Size | Target |
| --- | --- | --- |
| `app-arm64-v8a-release.apk` | ~30 MB | modern phones (use this) |
| `app-armeabi-v7a-release.apk` | ~25 MB | old 32-bit phones |
| `app-x86_64-release.apk` | ~33 MB | emulators |
| `app-release.apk` | ~69 MB | any phone, single file |

Install one with `adb install -r build/app/outputs/flutter-apk/app-arm64-v8a-release.apk`,
or copy it to the phone and open it.

The debug APK is ~220 MB because it ships a full Dart VM/JIT, kernel blob, and
all three ABIs. Never use it for distribution.

## Model asset

`assets/fruit_model.tflite` and `assets/labels.txt` are generated from the
trained Keras model (float16 weights, ~8 MB instead of ~16 MB):

```sh
cd ..
.venv\Scripts\python deep_learning\convert_to_tflite.py
```

Re-run it whenever `model_dl/efficientnet_b0_fruit_veg.h5` changes, then
rebuild the app.

`assets/detector.tflite` is Google's EfficientDet-Lite0 object detector
([source](https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite0_detection_metadata_1.tflite),
Apache-2.0). It proposes candidate boxes; the app classifies each cropped box
with the 36-class fruit model above.

## How it works

1. `CameraController.startImageStream` yields YUV420 frames.
2. Each frame is rotated upright using the camera sensor orientation, scaled to
   320x320 for the detector, and converted to RGB while sampling.
3. The detector proposes candidate boxes; only produce classes are kept.
   Overlapping boxes are removed with non-max suppression (up to three remain).
4. Each box is cropped from the upright frame, resized to 224x224, and
   classified. Pixels are raw 0-255 RGB: Keras 3's EfficientNetB0 contains its
   own rescaling/normalization layers, so no mean/std normalization is applied.
5. `tflite_flutter` runs both models; a green box and a `label confidence%` tag
   are drawn over each recognized fruit.

Only one frame is processed at a time (`_busy`), so inference never queues up.
