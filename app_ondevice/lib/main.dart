import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

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
  String _info = "Iniciando...";
  bool _listo = false;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    final camara = camaras.first; // cámara trasera
    _cam = CameraController(camara, ResolutionPreset.medium, enableAudio: false);
    _hands = HandLandmarkerPlugin.create(numHands: 2, delegate: HandLandmarkerDelegate.gpu);
    await _cam!.initialize();

    _hands!.landmarkStream.listen((hands) {
      if (!mounted) return;
      if (hands.isEmpty) {
        setState(() => _info = "Manos detectadas: 0");
      } else {
        final h = hands.first;
        final p0 = h.landmarks[0];   // muñeca
        final p8 = h.landmarks[8];   // punta del índice
        setState(() => _info =
          "Manos: ${hands.length}   |   puntos/mano: ${h.landmarks.length}\n"
          "Muñeca:  x=${p0.x.toStringAsFixed(3)}  y=${p0.y.toStringAsFixed(3)}\n"
          "Índice:  x=${p8.x.toStringAsFixed(3)}  y=${p8.y.toStringAsFixed(3)}");
      }
    });

    await _cam!.startImageStream((img) {
      _hands?.processFrame(img, _cam!.description.sensorOrientation);
    });
    if (mounted) setState(() => _listo = true);
  }

  @override
  void dispose() {
    _cam?.stopImageStream();
    _cam?.dispose();
    _hands?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Prueba keypoints - manos")),
      body: Column(children: [
        Expanded(child: (_listo && _cam != null)
            ? CameraPreview(_cam!)
            : const Center(child: CircularProgressIndicator())),
        Container(
          width: double.infinity, color: Colors.black,
          padding: const EdgeInsets.all(16),
          child: Text(_info, style: const TextStyle(color: Colors.greenAccent, fontSize: 18)),
        ),
      ]),
    );
  }
}