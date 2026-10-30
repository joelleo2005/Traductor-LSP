# Traductor LSP: reconocimiento de señas aisladas de la Lengua de Señas Peruana

Código fuente del trabajo *"Aplicativo móvil de traducción de palabras de la Lengua de Señas Peruana a texto en tiempo real"* (Universidad de Lima).

El sistema extrae los landmarks corporales de cada video con **MediaPipe Holistic** (33 puntos de la pose y 21 de cada mano, 258 valores por fotograma) y compara dos clasificadores convolucionales entrenados con **PyTorch** para reconocer **10 señas aisladas** de la LSP:

- **Conv1D** (modelo base): convoluciones en el tiempo sobre la secuencia de 48 × 258 valores.
- **Conv2D** (modelo propuesto): convoluciones sobre la imagen esqueleto-temporal **TSSI** (48 × 109 × 9).

---

## 1. Estructura del repositorio

```
Traductor-LSP/
├── README.md               # este archivo
├── DATASET.md              # enlaces públicos de los datasets (no se incluyen por su tamaño)
├── modelo/                 # PIPELINE DE INVESTIGACIÓN (extracción, entrenamiento, experimentos y resultados)
│   ├── requirements.txt
│   ├── data/               # se genera al ejecutar (landmarks, resultados, gráficos); vacío en el repositorio
│   └── src/
│       ├── procesar_todo.py     # Paso 1. Extrae los landmarks de los videos -> data/X.npy, data/y.npy
│       ├── keypoints.py         #         Salida de MediaPipe -> vector de 258 valores por fotograma
│       ├── eaf.py               #         Lectura de las anotaciones ELAN (.eaf) del dataset PUCP 305
│       ├── comun.py             # Base común: CONFIG, carga, partición, aumento, entrenamiento y evaluación
│       ├── normalizar.py        #         Normalización de los landmarks respecto a los hombros
│       ├── augmentation.py      #         Aumento de datos (ruido, escala, rotación, deformación temporal, espejo)
│       ├── experimentos.py      # Paso 2. Pruebas de la fase de ajuste (una semilla, un cambio a la vez)
│       ├── entrenar_conv1d.py   # Paso 3. Modelo base Conv1D: evaluación final en 5 semillas
│       ├── entrenar_conv2d.py   # Paso 3. Modelo propuesto Conv2D (TSSI): evaluación final en 5 semillas
│       ├── comparar_modelos.py  # Paso 4. Tabla comparativa en data/resultados.xlsx
│       └── graficos.py          # Paso 4. Gráficos PNG en data/graficos/
├── backend/                # Prototipo de servidor (FastAPI) que recibe un video y devuelve la seña
├── app/                    # Prototipo de app Flutter que graba la seña y la envía al backend
└── app_ondevice/           # Prototipo de app Flutter con inferencia en el propio celular (TFLite)
```

> El archivo `.zip` del código fuente **no incluye** entornos virtuales (`.venv/`), dependencias compiladas
> (`build/`, `.dart_tool/`), cachés (`__pycache__/`), configuraciones del IDE (`.idea/`) ni datos (`modelo/data/`).
> Todo ello se regenera con los pasos de este documento.

---

## 2. Prerrequisitos

| Componente | Versión | Observación |
|---|---|---|
| Python | **3.11** (probado con 3.11.9) | Otras versiones pueden no ser compatibles con las dependencias fijadas |
| PyTorch | 2.7.1 + CUDA 12.8 | Se instala con `requirements.txt` |
| MediaPipe | 0.10.14 (con `protobuf` 4.25.9) | Última versión con la API `mp.solutions.holistic` |
| NumPy / scikit-learn / OpenCV | 2.1.3 / 1.9.0 / 4.11.0.86 | Ver `modelo/requirements.txt` |
| matplotlib / openpyxl | 3.11.1 / 3.1.5 | Gráficos y Excel de resultados |
| Sistema operativo | Windows 10/11 o Linux | |
| GPU (opcional) | NVIDIA compatible con CUDA 12.8 | Sin GPU el entrenamiento corre en CPU, más lento |
| Flutter SDK (solo apps) | Dart ≥ 3.12 | Ver `environment` en cada `pubspec.yaml` |

Equipo con el que se obtuvieron los resultados reportados: Windows 11, Intel Core i7-14700K, 32 GB de RAM y NVIDIA GeForce RTX 5080.
Se necesitan unos 3 GB libres para el entorno virtual, más el espacio de los videos.

---

## 3. Instalación

Todos los comandos de las secciones 3 a 6 se ejecutan **desde la carpeta `modelo/`**.

### 3.1 Crear y activar el entorno virtual

**Windows (PowerShell):**

```powershell
cd modelo
py -3.11 -m venv .venv
.venv\Scripts\activate
```

> Si PowerShell indica que la ejecución de scripts está deshabilitada, ejecutar una vez
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y volver a activar.

**Linux / macOS:**

```bash
cd modelo
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3.2 Instalar las dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Comprobación:

```bash
python -c "import torch, mediapipe as mp; print(torch.__version__, torch.cuda.is_available(), mp.__version__)"
```

Debe mostrar `2.7.1+cu128 True 0.10.14` (`False` si el equipo no tiene GPU NVIDIA; en ese caso todo funciona en CPU).

> **Importante:** no actualizar `protobuf` ni `mediapipe` por separado; MediaPipe 0.10.14 necesita `protobuf<5`.

---

## 4. Datos

Los videos **no se incluyen** porque superan 1 GB y tienen licencias propias. Los enlaces públicos y las instrucciones
de descarga están en [`DATASET.md`](DATASET.md). Son tres datasets del Archivo Digital de Lenguas Peruanas (PUCP):

| Dataset | Enlace | Licencia |
|---|---|---|
| LSP - PUCP 305 (glosas) | https://hdl.handle.net/20.500.12534/JU4OLG | CC BY-NC 4.0 |
| LSP - Aprendo en Casa 2020 | https://hdl.handle.net/20.500.12534/HDOAGH | CC0 |
| LSP - PUCP-DGI156 | https://hdl.handle.net/20.500.12534/OJYYYS | CC0 |

Después de descargarlos y descomprimirlos, editar las tres rutas al inicio de `src/procesar_todo.py`:

```python
PERUSIL = r"C:\ruta\a\Videos\SEGMENTED_SIGN_ADJUSTED"                       # PUCP-DGI156
APRENDO = r"C:\ruta\a\2020-2021_LSP_peru1235_videos\Videos\SEGMENTED_SIGN"   # Aprendo en Casa 2020
PUCP    = r"C:\ruta\a\PUCP305_glosas\5. Segundo avance (corregido)"         # PUCP 305
```

---

## 5. Ejecución paso a paso

Con el entorno activado y desde `modelo/`:

### Paso 1. Extraer los landmarks (una sola vez)

```bash
python src/procesar_todo.py
```

Recorre los videos, extrae los landmarks de cada fotograma con MediaPipe Holistic, lleva cada seña a 48 fotogramas y guarda
`data/X.npy` (forma `(1331, 48, 258)`, 18 señas) y `data/y.npy`. Para entrenar se usan 10 de esas señas (946 muestras).
Es el paso más lento, porque MediaPipe analiza cada fotograma de cada video.

La línea base de los experimentos usa además una versión de 30 fotogramas: se obtiene ejecutando el mismo script con
`N_FRAMES = 30` y copiando `X.npy` y `y.npy` a la carpeta `data_base_30f/`. Solo es necesaria para el grupo `ajustes` del paso 2.

### Paso 2. Experimentos de la fase de ajuste (opcional)

```bash
python src/experimentos.py
```

Entrena los dos modelos con una sola partición (semilla 1), cambiando una cosa a la vez desde la línea base hasta la
configuración final. Guarda `data/experimentos/experimentos.csv` (una fila por prueba y modelo, con métricas de validación
y de prueba) y las curvas de cada prueba. Si se interrumpe, al volver a ejecutarlo continúa donde se quedó.

| Grupo (`--grupo`) | Contenido |
|---|---|
| `ajustes` | Línea base y ajustes 1 a 5 (más fotogramas, suavizado de etiquetas, corrección del aumento, 100 épocas) |
| `grilla1` | Suavizado × pesos por clase × copias de aumento (se elige con validación) |
| `grilla2` | Regularización L2 × dropout (se elige con validación) |
| `ajuste7` | Interpolación de las manos no detectadas y aumento con espejo |
| `todo` | Todos los grupos (valor por defecto) |

### Paso 3. Evaluación final (5 particiones)

```bash
python src/entrenar_conv1d.py
python src/entrenar_conv2d.py
```

Cada script entrena el modelo con `CONFIG` en 5 particiones distintas (semillas 1 a 5; 80 % entrenamiento / 20 % prueba,
estratificadas por seña) y muestra en la terminal el accuracy, el F1 macro, el reporte por seña y la matriz de confusión.
Ambos modelos usan las mismas semillas, por lo que se evalúan sobre las mismas particiones.

Archivos generados en `data/`: `resultados_<modelo>.csv`, `por_sena_<modelo>.csv`, `matriz_<modelo>.csv`,
`historial_<modelo>.csv` y `modelo_<modelo>.pt` (pesos del modelo de la primera semilla).

### Paso 4. Tabla comparativa y gráficos

```bash
python src/comparar_modelos.py
python src/graficos.py
```

- `data/resultados.xlsx`: resumen, resultados por semilla, métricas por seña, matrices de confusión e historial.
- `data/graficos/*.png`: accuracy y F1, F1 por seña, matrices de confusión, curvas de aprendizaje y costo computacional.

> Una ejecución nueva **reemplaza** los archivos anteriores con el mismo nombre en `data/`.

---

## 6. Configuración y ejemplos de uso

### 6.1 Configuración final (`CONFIG` en `src/comun.py`)

Se aplica por igual a los dos modelos.

| Parámetro | Valor | Descripción |
|---|---|---|
| `SEÑAS_USAR` | 10 señas | YO, PENSAR, MUJER, MAMA, QUE, VER, COMER, BIEN, CAMINAR, CASA |
| `fotogramas` | 48 | Fotogramas por seña |
| `epocas` | 100 | Número fijo de épocas (sin parada temprana) |
| `suavizado` | 0.05 | Suavizado de etiquetas (label smoothing) |
| `pesos_clase` | `False` | Pesos por clase en la función de pérdida |
| `copias_aug` | 8 | Copias aumentadas por cada seña de entrenamiento (entrenamiento × 9) |
| `ruido_en_ceros` | `False` | El ruido del aumento no mueve los puntos no detectados |
| `reg_l2` / `dropout` | 1e-4 / 0.5 | Regularización |
| `interpolar_manos` | `True` | Rellena los huecos interiores de las manos no detectadas por interpolación lineal |
| `espejo` | 0.5 | Probabilidad de reflejar cada copia aumentada (seña realizada con la otra mano) |
| `BATCH` / `LR` | 16 / 1e-3 | Tamaño de lote y tasa de aprendizaje inicial (Adam; se reduce a la mitad si la validación no mejora en 10 épocas) |
| `VAL_SIZE` | 0.15 | Fracción del 80 % de entrenamiento usada como validación |

El aumento de datos se aplica **después** de la partición y **solo** al conjunto de entrenamiento, por lo que no hay fuga de
datos hacia la validación ni la prueba.

### 6.2 Opciones de línea de comandos

`entrenar_conv1d.py` y `entrenar_conv2d.py`:

| Opción | Ejemplo | Efecto |
|---|---|---|
| `--semillas` | `--semillas 1 2 3` | Particiones a evaluar (por defecto 1 2 3 4 5) |
| `--espejo P` | `--espejo 0` | Cambia la probabilidad de espejo sin editar `CONFIG` |
| `--l2 X` | `--l2 0.001` | Cambia la regularización L2 sin editar `CONFIG` |
| `--sufijo S` | `--sufijo _sin_espejo` | Obligatorio con `--espejo` o `--l2`; los archivos se guardan como `resultados_<modelo><sufijo>.csv` para no reemplazar los de la configuración final |

`experimentos.py`:

| Opción | Ejemplo | Efecto |
|---|---|---|
| `--grupo` | `--grupo ajuste7` | Ejecuta solo un grupo de pruebas |
| `--modelos` | `--modelos conv2d` | Ejecuta solo un modelo (`conv1d`, `conv2d` o ambos) |
| `--semilla` | `--semilla 2` | Partición usada en las pruebas (por defecto 1) |

### 6.3 Ejemplos

```bash
# Prueba rápida: un solo modelo y una sola partición
python src/entrenar_conv2d.py --semillas 1

# Replicar la evaluación final completa de ambos modelos
python src/entrenar_conv1d.py --semillas 1 2 3 4 5
python src/entrenar_conv2d.py --semillas 1 2 3 4 5
python src/comparar_modelos.py
python src/graficos.py

# Medir el efecto del espejo: la misma evaluación sin aumento con espejo
python src/entrenar_conv2d.py --espejo 0 --sufijo _sin_espejo
#   -> data/resultados_conv2d_sin_espejo.csv (los resultados finales no se modifican)

# Probar una regularización L2 mayor
python src/entrenar_conv1d.py --l2 0.001 --sufijo _l2_1e-3

# Repetir solo las pruebas del ajuste 7 con el modelo propuesto
python src/experimentos.py --grupo ajuste7 --modelos conv2d
```

Para cambios que no tienen opción (por ejemplo, otras señas, otro número de épocas o desactivar la interpolación de manos),
se edita `SEÑAS_USAR` o `CONFIG` al inicio de `src/comun.py`.

### 6.4 Resultados de referencia

Con la configuración final y las semillas 1 a 5 (190 muestras de prueba por partición) se obtuvo:

| Modelo | Accuracy (%) | F1 macro (%) | Parámetros | Tamaño (KB) | Inferencia CPU (ms/seña) |
|---|---|---|---|---|---|
| Conv1D (base) | 75.8 ± 4.2 | 70.8 ± 4.7 | 83,978 | 328 | 2.72 |
| Conv2D (propuesto) | 79.1 ± 3.3 | 74.2 ± 3.2 | 170,030 | 664 | 1.78 |

Los entrenamientos son deterministas (`torch.backends.cudnn.deterministic = True`): con el mismo equipo, las mismas versiones
y la misma semilla se obtienen exactamente los mismos resultados. En otro hardware o con otras versiones de CUDA los valores
pueden variar ligeramente. Los tiempos de inferencia dependen del procesador.

---

## 7. Prototipos de backend y aplicaciones móviles

Estas carpetas contienen los prototipos de integración. Usan un modelo TFLite de una versión anterior del pipeline
(30 fotogramas y otro conjunto de 10 señas, ver `clases.txt`), todavía no actualizado a los modelos PyTorch finales.

**Backend (`backend/`)**, en un entorno virtual separado:

```bash
cd backend
pip install fastapi uvicorn python-multipart ai-edge-litert mediapipe==0.10.14 protobuf==4.25.9 opencv-python numpy
uvicorn servidor:app --host 0.0.0.0 --port 8000
```

`POST /predecir` recibe un video (campo `video`) y devuelve la seña reconocida y su confianza.

**App con servidor (`app/`)**: cambiar la IP en la constante `SERVIDOR` de `lib/main.dart` por la del equipo que ejecuta el backend y luego:

```bash
cd app
flutter pub get
flutter run
```

**App con inferencia en el celular (`app_ondevice/`)**: no necesita servidor; usa `assets/modelo_lsp.tflite`.

```bash
cd app_ondevice
flutter pub get
flutter run
```

---

## 8. Problemas frecuentes

| Problema | Solución |
|---|---|
| `AttributeError: ... has no attribute 'label'` o `'GetPrototype'` al usar MediaPipe | Se instaló una versión de `protobuf` demasiado nueva. Reinstalar con `pip install -r requirements.txt`. |
| `FileNotFoundError: ... X.npy` | Ejecutar primero `src/procesar_todo.py`, o copiar `X.npy` y `y.npy` en `modelo/data/`. |
| `ModuleNotFoundError: No module named 'comun'` | Ejecutar los comandos desde la carpeta `modelo/` con el entorno activado. |
| `torch.cuda.is_available()` devuelve `False` teniendo GPU | Actualizar el controlador de NVIDIA a uno compatible con CUDA 12.8. |
| `OSError` al guardar un gráfico | Cerrar la imagen PNG si está abierta en otro programa y volver a ejecutar `graficos.py`. |
