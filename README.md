# Traductor LSP

App móvil de traducción de Lengua de Señas Peruana (LSP) usando MediaPipe y Deep Learning. Reconocimiento de señas en el dispositivo (on-device).

## Estructura
- \modelo/\ — Pipeline en Python: extracción de landmarks (MediaPipe) y entrenamiento del modelo.
- \pp/\ — Aplicación móvil en Flutter que consume el modelo (.tflite).

## Dataset
PUCP 305 (glosas) - Lengua de Señas Peruana. Uso solo para investigación (CC BY-NC 4.0). Los videos NO se incluyen en el repo.
