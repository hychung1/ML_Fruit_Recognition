import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'detection.dart';

/// Draws a labelled rectangle for each recognized fruit on the camera preview.
class FruitBoxPainter extends CustomPainter {
  const FruitBoxPainter(this.detections);

  final List<Recognition> detections;

  @override
  void paint(Canvas canvas, Size size) {
    for (final detection in detections) {
      final rect = Rect.fromLTRB(
        detection.rect.left * size.width,
        detection.rect.top * size.height,
        detection.rect.right * size.width,
        detection.rect.bottom * size.height,
      );
      canvas.drawRect(
        rect,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3
          ..color = Colors.greenAccent,
      );

      final text = '${detection.label} ${(detection.confidence * 100).round()}%';
      final painter = TextPainter(
        text: TextSpan(
          text: text,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final tag = Rect.fromLTWH(
        rect.left,
        math.max(0, rect.top - 24),
        painter.width + 12,
        22,
      );
      canvas.drawRect(tag, Paint()..color = Colors.green);
      painter.paint(canvas, Offset(tag.left + 6, tag.top + 3));
    }
  }

  @override
  bool shouldRepaint(FruitBoxPainter oldDelegate) =>
      detections != oldDelegate.detections;
}
