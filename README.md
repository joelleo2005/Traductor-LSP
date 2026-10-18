
# Traductor LSP: reconocimiento de señas aisladas de la Lengua de Señas Peruana

Pipeline en Python que extrae landmarks corporales con **MediaPipe Holistic** a partir de videos de la Lengua de Señas Peruana (LSP) y entrena y compara dos clasificadores convolucionales, **Conv1D** (modelo base) y **Conv2D** (modelo propuesto, sobre una imagen esqueleto-temporal TSSI), para reconocer **10 señas aisladas**. Ambos modelos se exportan a **TensorFlow Lite**, el formato que usará la aplicación móvil (inferencia 100 % en el dispositivo).

---

## 1. Estructura

```
modelo/
├── README.md
├── requirements.txt
├── data/                  # se genera al ejecutar (no se incluye en el repositorio)
└── src/
    ├── procesar_todo.py   # 1. Extrae los landmarks de los videos -> data/X.npy, data/y.npy
    ├── keypoints.py       #    Convierte la salida de MediaPipe en un vector de 258 valores por fotograma
    ├── eaf.py             #    Lee las anotaciones ELAN (.eaf) de los videos de oraciones del PUCP
    ├── comun.py           # 2. Base común: carga, normalización, partición 80/20, aumento, entrenamiento y evaluación
    ├── normalizar.py      #    Normaliza los landmarks respecto a los hombros
    ├── augmentation.py    #    Aumento de datos (ruido, escala, rotación, deformación temporal)
    ├── entrenar_conv1d.py #    Modelo Conv1D (entrada 30 x 258)
    ├── entrenar_conv2d.py #    Modelo Conv2D (entrada 30 x 109 x 9, imagen esqueleto-temporal TSSI)
    ├── comparar_modelos.py# 3. Junta los resultados de ambos modelos en data/resultados.xlsx
    └── graficos.py        #    Genera los gráficos (PNG) a partir del Excel
```

---

## 2. Prerrequisitos

- **Python 3.11** (probado con 3.11.9). Otras versiones pueden no ser compatibles con las dependencias fijadas.
- Windows 10/11 o Linux.
- No se necesita GPU: el entrenamiento corre en CPU.
- Unos 3 GB libres para el entorno virtual y los datasets.

---

## 3. Instalación

Todos los comandos se ejecutan **desde la carpeta `modelo/`**.

### 3.1 Crear el entorno virtual

**Windows (PowerShell o terminal de Android Studio / VS Code):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

> Si PowerShell indica que la ejecución de scripts está deshabilitada, ejecutar una sola vez
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y volver a activar.

**Linux / macOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Al activarlo, la línea de la terminal empieza con `(.venv)`.

### 3.2 Instalar las dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para comprobar la instalación:

```bash
python -c "import tensorflow as tf, mediapipe as mp; print(tf.__version__, mp.__version__)"
```

Debe mostrar `2.19.1 0.10.14`.

> **Importante:** no actualizar `protobuf` ni `mediapipe` por separado. MediaPipe 0.10.14 (la última versión con la API
> `mp.solutions.holistic`) necesita `protobuf<5`. Por eso se fija TensorFlow 2.19.1, que es compatible con esa versión.

---

## 4. Datos

Los videos **no se incluyen** en el repositorio por su tamaño y su licencia. Se descargan del
[Archivo Digital de Lenguas Peruanas](https://datos.pucp.edu.pe/dataverse/adlp) (Portal de Datos Abiertos de la PUCP):

| Dataset | Enlace | Licencia |
|---|---|---|
| LSP - PUCP 305 (glosas) | https://hdl.handle.net/20.500.12534/JU4OLG | CC BY-NC 4.0 (solo investigación) |
| LSP - Aprendo En Casa 2020 | https://hdl.handle.net/20.500.12534/HDOAGH | CC0 |
| LSP - PUCP-DGI156 | https://hdl.handle.net/20.500.12534/OJYYYS | CC0 |

Después de descargarlos y descomprimirlos, **editar las tres rutas** al inicio de `src/procesar_todo.py`:

```python
PERUSIL = r"C:\ruta\a\Videos\SEGMENTED_SIGN_ADJUSTED"                         # PUCP-DGI156
APRENDO = r"C:\ruta\a\2020-2021_LSP_peru1235_videos\Videos\SEGMENTED_SIGN"     # Aprendo en Casa
PUCP    = r"C:\ruta\a\PUCP305_glosas\5. Segundo avance (corregido)"           # PUCP 305
```

> Si ya se cuenta con `data/X.npy` y `data/y.npy`, se pueden copiar en `modelo/data/` y omitir el paso 5.1.

---

## 5. Ejecución

Con el entorno activado y desde `modelo/`:

### 5.1 Extraer los landmarks (una sola vez)

```bash
python src/procesar_todo.py
```

Recorre los videos de los tres datasets, extrae con MediaPipe Holistic 33 puntos de la pose y 21 de cada mano por fotograma (258 valores), recorta cada seña a 30 fotogramas y guarda `data/X.npy` (forma `(1331, 30, 258)`) y `data/y.npy`.
Es el paso más lento del proceso, porque MediaPipe analiza cada fotograma de cada video.

### 5.2 Entrenar y evaluar los modelos

```bash
python src/entrenar_conv1d.py --semillas 1 2 3 4 5
python src/entrenar_conv2d.py --semillas 1 2 3 4 5
```

Cada script entrena el modelo en 5 particiones distintas (80 % entrenamiento / 20 % prueba, estratificadas por seña) y, para cada una, muestra el accuracy, el F1 macro, el reporte por seña y la matriz de confusión en la terminal.
Usar **las mismas semillas** en los dos modelos para que la comparación sea justa. Con una sola semilla: `--semillas 42`.

Genera en `data/`:

- `resultados_<modelo>.csv`, `por_sena_<modelo>.csv`, `matriz_<modelo>.csv`, `historial_<modelo>.csv`
- `modelo_<modelo>.keras` y `modelo_<modelo>.tflite` (modelo de la primera semilla)

### 5.3 Tabla comparativa y gráficos

```bash
python src/comparar_modelos.py
python src/graficos.py
```

- `data/resultados.xlsx`: resumen, resultados por semilla, métricas por seña, matrices de confusión e historial de entrenamiento.
- `data/graficos/*.png`: accuracy y F1, F1 por seña, matrices de confusión, curvas de aprendizaje por semilla y costo computacional.

> Las ejecuciones nuevas **reemplazan** los archivos anteriores de `data/`. Para conservar una corrida, copiar antes la carpeta.

---

## 6. Configuración

Los parámetros del experimento están al inicio de `src/comun.py` y se aplican por igual a los dos modelos:

| Parámetro | Valor | Descripción |
|---|---|---|
| `SEÑAS_USAR` | 10 señas | YO, PENSAR, MUJER, MAMA, QUE, VER, COMER, BIEN, CAMINAR, CASA |
| `EPOCHS` | 200 | Número fijo de épocas (sin early stopping) |
| `BATCH` | 16 | Tamaño de lote |
| `N_COPIAS_AUG` | 8 | Copias aumentadas por cada seña de entrenamiento |
| `VAL_SIZE` | 0.15 | Fracción del entrenamiento usada como validación |

El aumento de datos se aplica **después** de la partición y **solo** al conjunto de entrenamiento, por lo que no hay fuga de datos hacia la validación ni la prueba.

---

## 7. Problemas frecuentes

| Problema | Solución |
|---|---|
| `AttributeError: ... has no attribute 'label'` o `'GetPrototype'` al usar MediaPipe | Se instaló una versión de `protobuf` demasiado nueva. Reinstalar con `pip install -r requirements.txt`. |
| `FileNotFoundError: data/X.npy` | Ejecutar primero `src/procesar_todo.py`, o copiar `X.npy` y `y.npy` en `modelo/data/`. |
| `ModuleNotFoundError: No module named 'comun'` | Ejecutar los comandos desde la carpeta `modelo/` y con el entorno activado. |
| `ModuleNotFoundError: No module named 'openpyxl'` | `pip install -r requirements.txt` |
