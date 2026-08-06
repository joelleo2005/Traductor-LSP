import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

// >>> TU IP del servidor <
const String SERVIDOR = "http://192.168.1.105:8000/predecir";
late List<CameraDescription> camaras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  camaras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Traductor LSP',
      theme: ThemeData(primarySwatch: Colors.deepPurple),
      home: const PantallaCamara(),
    );
  }
}

class PantallaCamara extends StatefulWidget {
  const PantallaCamara({super.key});
  @override
  State<PantallaCamara> createState() => _PantallaCamaraState();
}

class _PantallaCamaraState extends State<PantallaCamara> {
  CameraController? _controller;
  bool _ocupado = false;
  String _resultado = "Presiona el botón y haz una seña";

  @override
  void initState() {
    super.initState();
    _iniciarCamara();
  }

  Future<void> _iniciarCamara() async {
    final camara = camaras.first; // cámara trasera (evita el espejo de la frontal)
    _controller = CameraController(camara, ResolutionPreset.medium, enableAudio: false);
    await _controller!.initialize();
    if (mounted) setState(() {});
  }

  Future<void> _grabarYEnviar() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    setState(() { _ocupado = true; _resultado = "Grabando..."; });

    await _controller!.startVideoRecording();
    await Future.delayed(const Duration(milliseconds: 2500)); // graba 2.5 seg
    final XFile video = await _controller!.stopVideoRecording();

    setState(() { _resultado = "Analizando..."; });

    try {
      final req = http.MultipartRequest("POST", Uri.parse(SERVIDOR));
      req.files.add(await http.MultipartFile.fromPath("video", video.path));
      final resp = await req.send();
      final cuerpo = await resp.stream.bytesToString();
      final datos = jsonDecode(cuerpo);
      final sena = datos["sena"];
      final conf = (datos["confianza"] * 100).toStringAsFixed(1);
      setState(() { _resultado = "$sena  ($conf%)"; });
    } catch (e) {
      setState(() { _resultado = "Error: no se pudo conectar al servidor"; });
    } finally {
      setState(() { _ocupado = false; });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Traductor LSP")),
      body: Column(
        children: [
          Expanded(
            child: (_controller != null && _controller!.value.isInitialized)
                ? CameraPreview(_controller!)
                : const Center(child: CircularProgressIndicator()),
          ),
          Container(
            padding: const EdgeInsets.all(24),
            color: Colors.black,
            width: double.infinity,
            child: Text(
              _resultado,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white, fontSize: 26, fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _ocupado ? null : _grabarYEnviar,
        icon: const Icon(Icons.sign_language),
        label: Text(_ocupado ? "Procesando..." : "Grabar seña"),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}