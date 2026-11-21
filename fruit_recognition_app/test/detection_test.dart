import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:fruit_recognition_app/detection.dart';

void main() {
  test('decodeDetections converts [top, left, bottom, right] and filters', () {
    final boxes = Float32List(kMaxDetections * 4);
    final scores = Float32List(kMaxDetections);
    final classes = Float32List(kMaxDetections);

    // Candidate 0: valid apple box [top .1, left .2, bottom .6, right .8].
    boxes.setAll(0, [0.1, 0.2, 0.6, 0.8]);
    scores[0] = 0.9;
    classes[0] = 52;
    // Candidate 1: low score, dropped.
    boxes.setAll(4, [0.0, 0.0, 1.0, 1.0]);
    scores[1] = 0.2;
    // Candidate 2: non-produce class, dropped.
    boxes.setAll(8, [0.7, 0.5, 0.3, 0.9]);
    scores[2] = 0.8;
    classes[2] = 1;

    final decoded = decodeDetections(boxes, scores, classes);
    expect(decoded, hasLength(1));
    expect(decoded.single.rect.left, closeTo(0.2, 1e-6));
    expect(decoded.single.rect.top, closeTo(0.1, 1e-6));
    expect(decoded.single.rect.right, closeTo(0.8, 1e-6));
    expect(decoded.single.rect.bottom, closeTo(0.6, 1e-6));
  });

  test('nonMaxSuppression keeps distinct boxes and drops duplicates', () {
    final boxes = [
      DetectedBox(Rect.fromLTRB(0.1, 0.1, 0.5, 0.5), 0.7),
      DetectedBox(Rect.fromLTRB(0.12, 0.12, 0.52, 0.52), 0.9),
      DetectedBox(Rect.fromLTRB(0.6, 0.6, 0.9, 0.9), 0.6),
    ];

    final picked = nonMaxSuppression(boxes, maxBoxes: 3);
    expect(picked, hasLength(2));
    expect(picked.first.score, 0.9);
    expect(picked.last.rect, Rect.fromLTRB(0.6, 0.6, 0.9, 0.9));
  });

  test('intersectionOverUnion is 1 for identical boxes and 0 for separate', () {
    final a = Rect.fromLTRB(0, 0, 1, 1);
    expect(intersectionOverUnion(a, a), 1);
    expect(
      intersectionOverUnion(a, Rect.fromLTRB(2, 2, 3, 3)),
      0,
    );
  });
}
