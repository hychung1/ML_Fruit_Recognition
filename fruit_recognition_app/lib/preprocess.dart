import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:camera/camera.dart';

const kImageSize = 224;
const kDetectorSize = 320;

/// Reads YUV420 camera pixels into upright RGB, writing one value per pixel.
///
/// [region] is an optional normalized rectangle of the upright frame, used to
/// crop a detected box before classification. [rotation] is the clockwise
/// rotation that makes the frame upright.
void _sampleRgb(
  CameraImage image,
  int rotation,
  int size,
  Rect? region,
  void Function(int pixel, num r, num g, num b) write,
) {
  final planes = image.planes;
  if (planes.length < 3) {
    throw ArgumentError('Expected a YUV420 frame with 3 planes');
  }

  final w = image.width;
  final h = image.height;
  final yPlane = planes[0];
  final uPlane = planes[1];
  final vPlane = planes[2];
  final yRowStride = yPlane.bytesPerRow;
  final yPixelStride = yPlane.bytesPerPixel ?? 1;
  final uRowStride = uPlane.bytesPerRow;
  final uPixelStride = uPlane.bytesPerPixel ?? 1;
  final vRowStride = vPlane.bytesPerRow;
  final vPixelStride = vPlane.bytesPerPixel ?? 1;

  // Size of the frame after rotating it upright.
  final rotatedWidth = rotation % 180 == 0 ? w : h;
  final rotatedHeight = rotation % 180 == 0 ? h : w;
  final left = (region?.left ?? 0) * rotatedWidth;
  final top = (region?.top ?? 0) * rotatedHeight;
  final regionWidth = (region?.width ?? 1) * rotatedWidth;
  final regionHeight = (region?.height ?? 1) * rotatedHeight;

  var p = 0;
  for (var oy = 0; oy < size; oy++) {
    final ry = (top + (oy + 0.5) * regionHeight / size)
        .floor()
        .clamp(0, rotatedHeight - 1);
    for (var ox = 0; ox < size; ox++) {
      final rx = (left + (ox + 0.5) * regionWidth / size)
          .floor()
          .clamp(0, rotatedWidth - 1);
      // Map the upright pixel back to the raw sensor pixel.
      final (sx, sy) = switch (rotation) {
        90 => (ry, h - 1 - rx),
        180 => (w - 1 - rx, h - 1 - ry),
        270 => (w - 1 - ry, rx),
        _ => (rx, ry),
      };

      final y = yPlane.bytes[sy * yRowStride + sx * yPixelStride];
      final u =
          uPlane.bytes[(sy >> 1) * uRowStride + (sx >> 1) * uPixelStride] - 128;
      final v =
          vPlane.bytes[(sy >> 1) * vRowStride + (sx >> 1) * vPixelStride] - 128;

      write(
        p++,
        y + 1.402 * v,
        y - 0.344136 * u - 0.714136 * v,
        y + 1.772 * u,
      );
    }
  }
}

/// Raw 0-255 RGB floats for the 36-class EfficientNet classifier.
///
/// Keras 3's EfficientNetB0 rescales internally, so no mean/std normalization
/// is applied here. A [region] crops one detected box out of the frame.
void yuvToModelInput(
  Float32List out,
  CameraImage image,
  int rotation, [
  Rect? region,
]) {
  _sampleRgb(image, rotation, kImageSize, region, (p, r, g, b) {
    final i = p * 3;
    out[i] = r.clamp(0.0, 255.0).toDouble();
    out[i + 1] = g.clamp(0.0, 255.0).toDouble();
    out[i + 2] = b.clamp(0.0, 255.0).toDouble();
  });
}

/// Raw 0-255 RGB bytes for the quantized EfficientDet detector.
void yuvToDetectorInput(Uint8List out, CameraImage image, int rotation) {
  _sampleRgb(image, rotation, kDetectorSize, null, (p, r, g, b) {
    final i = p * 3;
    out[i] = r.clamp(0, 255).toInt();
    out[i + 1] = g.clamp(0, 255).toInt();
    out[i + 2] = b.clamp(0, 255).toInt();
  });
}
