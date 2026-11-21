import 'dart:typed_data';
import 'dart:ui' show Rect;

/// EfficientDet-Lite0 returns at most 25 candidate boxes.
const kMaxDetections = 25;

/// COCO produce class indices (model output is COCO id - 1): banana, apple,
/// orange, broccoli, carrot. Other objects are ignored before classification.
const kFruitClassIds = {51, 52, 54, 55, 56};

class DetectedBox {
  const DetectedBox(this.rect, this.score);

  /// Normalized (0..1) box in the upright camera frame.
  final Rect rect;
  final double score;
}

class Recognition {
  const Recognition(this.rect, this.label, this.confidence);

  final Rect rect;
  final String label;
  final double confidence;
}

double _clamp01(double v) => v < 0 ? 0 : v > 1 ? 1 : v;

/// Converts the detector's [yMin, xMin, yMax, xMax] output into normalized
/// rectangles and drops non-produce, low-confidence or malformed boxes.
List<DetectedBox> decodeDetections(
  Float32List boxes,
  Float32List scores,
  Float32List classes, {
  double minScore = 0.25,
}) {
  final result = <DetectedBox>[];
  for (var i = 0; i < kMaxDetections; i++) {
    final score = scores[i];
    if (score < minScore) continue;
    if (!kFruitClassIds.contains(classes[i].round())) continue;
    final top = _clamp01(boxes[i * 4]);
    final left = _clamp01(boxes[i * 4 + 1]);
    final bottom = _clamp01(boxes[i * 4 + 2]);
    final right = _clamp01(boxes[i * 4 + 3]);
    if (right <= left || bottom <= top) continue;
    result.add(DetectedBox(Rect.fromLTRB(left, top, right, bottom), score));
  }
  return result;
}

double intersectionOverUnion(Rect a, Rect b) {
  final overlap = a.intersect(b);
  if (overlap.isEmpty) return 0;
  final area = a.width * a.height + b.width * b.height;
  final intersection = overlap.width * overlap.height;
  return intersection / (area - intersection);
}

/// Keeps the highest-scoring boxes, suppressing duplicates that overlap.
List<DetectedBox> nonMaxSuppression(
  List<DetectedBox> boxes, {
  double iouThreshold = 0.5,
  int maxBoxes = 3,
}) {
  final sorted = [...boxes]..sort((a, b) => b.score.compareTo(a.score));
  final picked = <DetectedBox>[];
  for (final box in sorted) {
    if (picked.any(
      (other) => intersectionOverUnion(box.rect, other.rect) >= iouThreshold,
    )) {
      continue;
    }
    picked.add(box);
    if (picked.length == maxBoxes) break;
  }
  return picked;
}
