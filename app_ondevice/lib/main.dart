import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:flutter_pose_detection/flutter_pose_detection.dart';

late List<CameraDescription> camaras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  camaras = await availableCameras();
  runApp(const MaterialApp(home: PruebaKeypoints()));
}

class PruebaKeypoints extends StatefulWidget {
  const PruebaKeypoints({super.key});
  @override
  State<PruebaKeypoints> createState() => _PruebaKeypointsState();
}

class _PruebaKeypointsState extends State<PruebaKeypoints> {
  CameraController? _cam;
  HandLandmarkerPlugin? _hands;
  NpuPoseDetector? _pose;
  bool _procesandoPose = false;
  bool _listo = false;
  int _numManos = 0;
  String _infoPose = "Pose: -";

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    final camara = camaras.first;
    _cam = CameraController(camara, ResolutionPreset.medium, enableAudio: false);
    _hands = HandLandmarkerPlugin.create(numHands: 2, delegate: HandLandmarkerDelegate.gpu);
    _pose = NpuPoseDetector(config: PoseDetectorConfig.realtime());
    await _pose!.initialize();
    await _cam!.initialize();

    _hands!.landmarkStream.listen((hands) {
      _numManos = hands.length;
      if (mounted) setState(() {});
    });

    await _cam!.startImageStream(_procesarFrame);
    if (mounted) setState(() => _listo = true);
  }

  Future<void> _procesarFrame(CameraImage image) async {
    _hands?.processFrame(image, _cam!.description.sensorOrientation);

    if (_procesandoPose) return;  // throttle: no encolar frames
    _procesandoPose = true;
    try {
      final planes = image.planes.map((p) => {
        'bytes': p.bytes,
        'bytesPerRow': p.bytesPerRow,
        'bytesPerPixel': p.bytesPerPixel,
      }).toList();
      final result = await _pose!.processFrame(
        planes: planes,
        width: image.width,
        height: image.height,
        format: 'yuv420',
        rotation: _cam!.description.sensorOrientation,
      );
      if (result.hasPoses) {
        final n = result.firstPose!.landmarks.first;
        _infoPose = "Pose: ${result.firstPose!.landmarks.length} pts  (nariz x=${n.x.toStringAsFixed(2)} y=${n.y.toStringAsFixed(2)})";
      } else {
        _infoPose = "Pose: 0";
      }
      if (mounted) setState(() {});
    } catch (e) {
      _infoPose = "Pose error: $e";
    } finally {
      _procesandoPose = false;
    }
  }

  @override
  void dispose() {
    _cam?.stopImageStream();
    _cam?.dispose();
    _hands?.dispose();
    _pose?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Prueba manos + pose")),
      body: Column(children: [
        Expanded(child: (_listo && _cam != null)
            ? CameraPreview(_cam!)
            : const Center(child: CircularProgressIndicator())),
        Container(
          width: double.infinity, color: Colors.black,
          padding: const EdgeInsets.all(16),
          child: Text(
            "Manos: $_numManos (21 c/u)\n$_infoPose",
            style: const TextStyle(color: Colors.greenAccent, fontSize: 18)),
        ),
      ]),
    );
  }
}