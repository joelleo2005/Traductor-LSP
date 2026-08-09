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

  List<Hand> _ultimasManos = [];
  dynamic _ultimaPose;
  String _info = "Iniciando...";

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
      _ultimasManos = hands;
    });

    await _cam!.startImageStream(_procesarFrame);
    if (mounted) setState(() => _listo = true);
  }

  Future<void> _procesarFrame(CameraImage image) async {
    _hands?.processFrame(image, _cam!.description.sensorOrientation);
    if (_procesandoPose) return;
    _procesandoPose = true;
    try {
      final planes = image.planes.map((p) => {
        'bytes': p.bytes,
        'bytesPerRow': p.bytesPerRow,
        'bytesPerPixel': p.bytesPerPixel,
      }).toList();
      final result = await _pose!.processFrame(
          planes: planes, width: image.width, height: image.height,
          format: 'yuv420', rotation: _cam!.description.sensorOrientation);
      _ultimaPose = result.hasPoses ? result.firstPose : null;

      final v = _construir258();
      if (mounted) setState(() {
        _info = "Vector: ${v.length} valores\n"
            "Manos: ${_ultimasManos.length}   Pose: ${_ultimaPose != null ? 33 : 0}\n"
            "muestra: ${v[0].toStringAsFixed(2)}, ${v[132].toStringAsFixed(2)}, ${v[195].toStringAsFixed(2)}";
      });
    } catch (e) {
      _info = "Error: $e";
    } finally {
      _procesandoPose = false;
    }
  }

  // Construye el vector de 258: pose(132) + mano_izq(63) + mano_der(63)
  List<double> _construir258() {
    final v = <double>[];
    // POSE: 33 x (x,y,z,visibility)
    if (_ultimaPose != null) {
      for (final l in _ultimaPose.landmarks) {
        v.addAll([(l.x as num).toDouble(), (l.y as num).toDouble(),
          (l.z as num).toDouble(), (l.visibility as num).toDouble()]);
      }
    } else {
      v.addAll(List.filled(132, 0.0));
    }
    // MANOS: asignar izquierda/derecha por posición en x
    List<double> aVec(Hand h) {
      final r = <double>[];
      for (final l in h.landmarks) { r.addAll([l.x, l.y, l.z]); }
      return r;
    }
    double promX(Hand h) => h.landmarks.map((l) => l.x).reduce((a, b) => a + b) / h.landmarks.length;

    List<double> izq = List.filled(63, 0.0), der = List.filled(63, 0.0);
    if (_ultimasManos.length == 1) {
      if (promX(_ultimasManos[0]) < 0.5) izq = aVec(_ultimasManos[0]);
      else der = aVec(_ultimasManos[0]);
    } else if (_ultimasManos.length >= 2) {
      final ord = [..._ultimasManos]..sort((a, b) => promX(a).compareTo(promX(b)));
      izq = aVec(ord[0]);
      der = aVec(ord[1]);
    }
    v.addAll(izq);
    v.addAll(der);
    return v; // 258
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
      appBar: AppBar(title: const Text("Vector 258")),
      body: Column(children: [
        Expanded(child: (_listo && _cam != null)
            ? CameraPreview(_cam!)
            : const Center(child: CircularProgressIndicator())),
        Container(width: double.infinity, color: Colors.black, padding: const EdgeInsets.all(16),
            child: Text(_info, style: const TextStyle(color: Colors.greenAccent, fontSize: 16))),
      ]),
    );
  }
}