import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:camera/camera.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:flutter_pose_detection/flutter_pose_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:path_provider/path_provider.dart';

late List<CameraDescription> camaras;
const SENAS = ["YO","VER","CAMINAR","CASA","PENSAR","MAMA","MUJER","HOMBRE","ESPERAR","QUE"];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  camaras = await availableCameras();
  runApp(const MaterialApp(home: Menu()));
}

List<double> construir258(List<Hand> manos, dynamic pose) {
  final v = <double>[];
  if (pose != null) {
    for (final l in pose.landmarks) {
      v.addAll([(l.x as num).toDouble(), (l.y as num).toDouble(),
        (l.z as num).toDouble(), (l.visibility as num).toDouble()]);
    }
  } else { v.addAll(List.filled(132, 0.0)); }
  List<double> aVec(Hand h) { final r = <double>[]; for (final l in h.landmarks) { r.addAll([l.x, l.y, l.z]); } return r; }
  double promX(Hand h) => h.landmarks.map((l) => l.x).reduce((a, b) => a + b) / h.landmarks.length;
  List<double> izq = List.filled(63, 0.0), der = List.filled(63, 0.0);
  if (manos.length == 1) {
    if (promX(manos[0]) < 0.5) izq = aVec(manos[0]); else der = aVec(manos[0]);
  } else if (manos.length >= 2) {
    final ord = [...manos]..sort((a, b) => promX(a).compareTo(promX(b)));
    izq = aVec(ord[0]); der = aVec(ord[1]);
  }
  v.addAll(izq); v.addAll(der);
  return v;
}

List<List<double>> ajustar30(List<List<double>> f) {
  if (f.isEmpty) return List.generate(30, (_) => List.filled(258, 0.0));
  final n = f.length;
  return List.generate(30, (i) => f[((i * (n - 1)) / 29).round()]);
}

class Menu extends StatelessWidget {
  const Menu({super.key});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Traductor LSP")),
      body: Center(child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        ElevatedButton.icon(icon: const Icon(Icons.sign_language), label: const Text("Reconocer"),
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const PantallaReconocer()))),
        const SizedBox(height: 20),
        ElevatedButton.icon(icon: const Icon(Icons.fiber_manual_record), label: const Text("Grabar datos"),
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const PantallaGrabar()))),
      ])),
    );
  }
}

// ---------------- PANTALLA GRABAR ----------------
class PantallaGrabar extends StatefulWidget {
  const PantallaGrabar({super.key});
  @override
  State<PantallaGrabar> createState() => _PantallaGrabarState();
}

class _PantallaGrabarState extends State<PantallaGrabar> {
  CameraController? _cam;
  HandLandmarkerPlugin? _hands;
  NpuPoseDetector? _pose;
  bool _procesando = false, _listo = false;
  List<Hand> _manos = [];
  dynamic _ultimaPose;
  final List<List<double>> _buffer = [];
  String _sena = "YO";
  int _conteo = 0;
  String _estado = "Elige seña y toca Grabar";

  @override
  void initState() { super.initState(); _init(); }

  Future<void> _init() async {
    _cam = CameraController(camaras.first, ResolutionPreset.medium, enableAudio: false);
    _hands = HandLandmarkerPlugin.create(numHands: 2, delegate: HandLandmarkerDelegate.gpu);
    _pose = NpuPoseDetector(config: PoseDetectorConfig.realtime());
    await _pose!.initialize();
    await _cam!.initialize();
    _hands!.landmarkStream.listen((h) { _manos = h; });
    await _cam!.startImageStream(_frame);
    if (mounted) setState(() => _listo = true);
  }

  Future<void> _frame(CameraImage image) async {
    _hands?.processFrame(image, _cam!.description.sensorOrientation);
    if (_procesando) return;
    _procesando = true;
    try {
      final planes = image.planes.map((p) => {'bytes': p.bytes, 'bytesPerRow': p.bytesPerRow, 'bytesPerPixel': p.bytesPerPixel}).toList();
      final r = await _pose!.processFrame(planes: planes, width: image.width, height: image.height, format: 'yuv420', rotation: _cam!.description.sensorOrientation);
      _ultimaPose = r.hasPoses ? r.firstPose : null;
      _buffer.add(construir258(_manos, _ultimaPose));
      if (_buffer.length > 60) _buffer.removeAt(0);
    } catch (e) {} finally { _procesando = false; }
  }

  Future<void> _grabar() async {
    _buffer.clear();
    setState(() => _estado = "Grabando $_sena...");
    await Future.delayed(const Duration(milliseconds: 2500));
    if (_buffer.length < 5) { setState(() => _estado = "No te detecté, reintenta"); return; }
    final seq = ajustar30(_buffer);
    final dir = await getExternalStorageDirectory();
    final f = File('${dir!.path}/grabaciones.csv');
    final linea = "$_sena," + seq.expand((row) => row).map((v) => v.toStringAsFixed(5)).join(",");
    await f.writeAsString("$linea\n", mode: FileMode.append);
    _conteo++;
    setState(() => _estado = "Guardado ✓  ($_sena, total: $_conteo)");
  }

  @override
  void dispose() { _cam?.stopImageStream(); _cam?.dispose(); _hands?.dispose(); _pose?.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Grabar datos")),
      body: Column(children: [
        Expanded(child: (_listo && _cam != null) ? CameraPreview(_cam!) : const Center(child: CircularProgressIndicator())),
        Container(color: Colors.black, padding: const EdgeInsets.all(12), width: double.infinity, child: Column(children: [
          DropdownButton<String>(dropdownColor: Colors.black, value: _sena,
              items: SENAS.map((s) => DropdownMenuItem(value: s, child: Text(s, style: const TextStyle(color: Colors.white)))).toList(),
              onChanged: (v) => setState(() => _sena = v!)),
          Text(_estado, style: const TextStyle(color: Colors.greenAccent, fontSize: 18)),
        ])),
      ]),
      floatingActionButton: FloatingActionButton.extended(
          onPressed: _listo ? _grabar : null, icon: const Icon(Icons.fiber_manual_record), label: const Text("Grabar")),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}

// ---------------- PANTALLA RECONOCER ----------------
class PantallaReconocer extends StatefulWidget {
  const PantallaReconocer({super.key});
  @override
  State<PantallaReconocer> createState() => _PantallaReconocerState();
}

class _PantallaReconocerState extends State<PantallaReconocer> {
  CameraController? _cam;
  HandLandmarkerPlugin? _hands;
  NpuPoseDetector? _pose;
  Interpreter? _modelo;
  List<String> _clases = [];
  bool _procesando = false, _listo = false;
  List<Hand> _manos = [];
  dynamic _ultimaPose;
  final List<List<double>> _buffer = [];
  String _resultado = "Toca Reconocer y haz una seña";

  @override
  void initState() { super.initState(); _init(); }

  Future<void> _init() async {
    _cam = CameraController(camaras.first, ResolutionPreset.medium, enableAudio: false);
    _hands = HandLandmarkerPlugin.create(numHands: 2, delegate: HandLandmarkerDelegate.gpu);
    _pose = NpuPoseDetector(config: PoseDetectorConfig.realtime());
    await _pose!.initialize();
    _modelo = await Interpreter.fromAsset('assets/modelo_lsp.tflite');
    final txt = await rootBundle.loadString('assets/clases.txt');
    _clases = txt.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();
    await _cam!.initialize();
    _hands!.landmarkStream.listen((h) { _manos = h; });
    await _cam!.startImageStream(_frame);
    if (mounted) setState(() => _listo = true);
  }

  Future<void> _frame(CameraImage image) async {
    _hands?.processFrame(image, _cam!.description.sensorOrientation);
    if (_procesando) return;
    _procesando = true;
    try {
      final planes = image.planes.map((p) => {'bytes': p.bytes, 'bytesPerRow': p.bytesPerRow, 'bytesPerPixel': p.bytesPerPixel}).toList();
      final r = await _pose!.processFrame(planes: planes, width: image.width, height: image.height, format: 'yuv420', rotation: _cam!.description.sensorOrientation);
      _ultimaPose = r.hasPoses ? r.firstPose : null;
      _buffer.add(construir258(_manos, _ultimaPose));
      if (_buffer.length > 60) _buffer.removeAt(0);
    } catch (e) {} finally { _procesando = false; }
  }

  List<double> _normalizar(List<double> f) {
    final v = List<double>.from(f);
    final lx = v[44], ly = v[45], rx = v[48], ry = v[49];
    if (lx == 0 && rx == 0) return v;
    final cx = (lx + rx) / 2, cy = (ly + ry) / 2;
    var esc = sqrt((lx - rx) * (lx - rx) + (ly - ry) * (ly - ry));
    if (esc == 0) esc = 1.0;
    for (int i = 0; i < 33; i++) { v[4 * i] = (v[4 * i] - cx) / esc; v[4 * i + 1] = (v[4 * i + 1] - cy) / esc; }
    for (int b = 132; b < 258; b += 3) { v[b] = (v[b] - cx) / esc; v[b + 1] = (v[b + 1] - cy) / esc; }
    return v;
  }

  Future<void> _reconocer() async {
    if (_modelo == null) return;
    _buffer.clear();
    setState(() => _resultado = "Grabando...");
    await Future.delayed(const Duration(milliseconds: 2500));
    setState(() => _resultado = "Analizando...");
    if (_buffer.length < 5) { setState(() => _resultado = "No te detecté, reintenta"); return; }
    final seq = ajustar30(_buffer).map(_normalizar).toList();
    final output = List.filled(_clases.length, 0.0).reshape([1, _clases.length]);
    _modelo!.run([seq], output);
    final probs = (output[0] as List).cast<double>();
    int idx = 0; double best = probs[0];
    for (int i = 1; i < probs.length; i++) { if (probs[i] > best) { best = probs[i]; idx = i; } }
    setState(() => _resultado = "${_clases[idx]}   (${(best * 100).toStringAsFixed(0)}%)");
  }

  @override
  void dispose() { _cam?.stopImageStream(); _cam?.dispose(); _hands?.dispose(); _pose?.dispose(); _modelo?.close(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Reconocer")),
      body: Column(children: [
        Expanded(child: (_listo && _cam != null) ? CameraPreview(_cam!) : const Center(child: CircularProgressIndicator())),
        Container(width: double.infinity, color: Colors.black, padding: const EdgeInsets.all(20),
            child: Text(_resultado, textAlign: TextAlign.center, style: const TextStyle(color: Colors.greenAccent, fontSize: 24, fontWeight: FontWeight.bold))),
      ]),
      floatingActionButton: FloatingActionButton.extended(onPressed: _listo ? _reconocer : null, icon: const Icon(Icons.sign_language), label: const Text("Reconocer")),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}