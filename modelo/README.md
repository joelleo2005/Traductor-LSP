# modelo/: pipeline de investigación (PyTorch)

Extracción de landmarks con MediaPipe Holistic, entrenamiento y comparación de los modelos Conv1D (base) y Conv2D sobre la
imagen esqueleto-temporal TSSI (propuesto) para 10 señas aisladas de la LSP.

Las instrucciones completas (prerrequisitos, instalación, datos, ejecución, configuración y ejemplos) están en el
[README principal](../README.md). Resumen, desde esta carpeta y con el entorno activado:

```bash
pip install -r requirements.txt
python src/procesar_todo.py        # 1. landmarks -> data/X.npy, data/y.npy (editar antes las rutas de los videos)
python src/experimentos.py         # 2. pruebas de la fase de ajuste (opcional)
python src/entrenar_conv1d.py      # 3. evaluación final del modelo base (semillas 1 a 5)
python src/entrenar_conv2d.py      # 3. evaluación final del modelo propuesto (semillas 1 a 5)
python src/comparar_modelos.py     # 4. data/resultados.xlsx
python src/graficos.py             # 4. data/graficos/*.png
```
