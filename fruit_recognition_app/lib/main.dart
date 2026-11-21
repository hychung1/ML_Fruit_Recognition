import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'box_painter.dart';
import 'detection.dart';
import 'preprocess.dart';

void main() => runApp(const FruitScannerApp());

/// Interprets a float32 tensor's raw bytes as a flat list.
Float32List _floats(Uint8List bytes) =>
    bytes.buffer.asFloat32List(bytes.offsetInBytes, bytes.lengthInBytes >> 2);

class FruitScannerApp extends StatelessWidget {
  const FruitScannerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fruit Scanner',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const ScannerPage(),
    );
  }
}

class ScannerPage extends StatefulWidget {
  const ScannerPage({super.key});

  @override
  State<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage>
    with WidgetsBindingObserver {
  CameraController? _camera;
  Interpreter? _interpreter;
  Interpreter? _detector;
  List<String> _labels = const [];
  Float32List? _input;
  Uint8List? _detectorInput;
  int _rotation = 90;

  String? _status = 'Starting camera...';
  String? _error;
  List<Recognition> _detections = const [];
  bool _busy = false;
  bool _disposed = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _start();
  }

  Future<void> _start() async {
    try {
      final labels = (await rootBundle.loadString('assets/labels.txt'))
          .split('\n')
          .map((l) => l.trim())
          .where((l) => l.isNotEmpty)
          .toList();
      final interpreter = await Interpreter.fromAsset(
        'assets/fruit_model.tflite',
        options: InterpreterOptions()..threads = 4,
      );
      final detector = await Interpreter.fromAsset(
        'assets/detector.tflite',
        options: InterpreterOptions()..threads = 4,
      );
      final cameras = await availableCameras();
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      if (_disposed) {
        controller.dispose();
        interpreter.close();
        detector.close();
        return;
      }

      _interpreter?.close();
      _detector?.close();
      setState(() {
        _camera = controller;
        _interpreter = interpreter;
        _detector = detector;
        _labels = labels;
        _input = Float32List(kImageSize * kImageSize * 3);
        _detectorInput = Uint8List(kDetectorSize * kDetectorSize * 3);
        // The app is locked to portrait, so the back camera only needs its
        // sensor rotation to become upright.
        _rotation = camera.sensorOrientation % 360;
        _detections = const [];
        _status = null;
        _error = null;
      });
      await controller.startImageStream(_onFrame);
    } catch (e) {
      if (_disposed) return;
      setState(() {
        _camera = null;
        _status = null;
        _error = 'Camera or model failed to start.\n$e';
      });
    }
  }

  void _onFrame(CameraImage image) {
    if (_busy || _disposed || _input == null || _detector == null) return;
    _busy = true;
    _processFrame(image).whenComplete(() => _busy = false);
  }

  Future<void> _processFrame(CameraImage image) async {
    final detector = _detector;
    final interpreter = _interpreter;
    final input = _input;
    final detectorInput = _detectorInput;
    if (detector == null ||
        interpreter == null ||
        input == null ||
        detectorInput == null ||
        image.planes.length < 3) {
      return;
    }

    try {
      yuvToDetectorInput(detectorInput, image, _rotation);
      detector.getInputTensor(0).data = detectorInput;
      detector.invoke();
      final boxes = _floats(detector.getOutputTensor(0).data);
      final classes = _floats(detector.getOutputTensor(1).data);
      final scores = _floats(detector.getOutputTensor(2).data);

      final picked = nonMaxSuppression(
        decodeDetections(boxes, scores, classes),
        maxBoxes: 3,
      );
      final recognitions = <Recognition>[];
      for (final box in picked) {
        yuvToModelInput(input, image, _rotation, box.rect);
        interpreter.getInputTensor(0).data = input.buffer.asUint8List();
        interpreter.invoke();
        final probs = _floats(interpreter.getOutputTensor(0).data);

        var best = 0;
        for (var i = 1; i < probs.length; i++) {
          if (probs[i] > probs[best]) best = i;
        }
        if (probs[best] < 0.35 || best >= _labels.length) continue;
        recognitions.add(Recognition(box.rect, _labels[best], probs[best]));
      }

      if (mounted && !_disposed) {
        setState(() => _detections = recognitions);
      }
    } catch (_) {
      // A dropped or malformed frame is normal while streaming; skip it.
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final camera = _camera;
    if (camera == null || !camera.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _camera = null;
      camera.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _start();
    }
  }

  @override
  void dispose() {
    _disposed = true;
    WidgetsBinding.instance.removeObserver(this);
    _camera?.dispose();
    _interpreter?.close();
    _detector?.close();
    super.dispose();
  }

  String _title(String label) =>
      label.isEmpty ? label : label[0].toUpperCase() + label.substring(1);

  @override
  Widget build(BuildContext context) {
    final camera = _camera;
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (camera != null && camera.value.isInitialized)
              Center(
                child: CameraPreview(
                  camera,
                  child: CustomPaint(painter: FruitBoxPainter(_detections)),
                ),
              )
            else
              _placeholder(),
            Positioned(
              top: 16,
              left: 16,
              right: 16,
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 14,
                    vertical: 8,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Text(
                    'Fruit & Vegetable Scanner',
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                ),
              ),
            ),
            Positioned(
              left: 16,
              right: 16,
              bottom: 24,
              child: _resultCard(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _placeholder() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_error == null) ...[
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(
                _status ?? '',
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white70),
              ),
            ] else ...[
              const Icon(Icons.no_photography, color: Colors.white54, size: 48),
              const SizedBox(height: 16),
              Text(
                _error!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white70),
              ),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: () {
                  setState(() => _error = null);
                  _start();
                },
                child: const Text('Retry'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _resultCard() {
    final detections = _detections;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            detections.isEmpty
                ? 'Point at fruits or vegetables'
                : '${detections.length} detected',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          for (final detection in detections) ...[
            const SizedBox(height: 4),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  _title(detection.label),
                  style: const TextStyle(color: Colors.white),
                ),
                Text(
                  '${(detection.confidence * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(color: Colors.white70),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}
