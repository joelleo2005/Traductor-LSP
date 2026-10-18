"""
Paso 1 del pipeline: extracción de landmarks de los videos.

Recorre los videos de los tres datasets, pasa cada fotograma por MediaPipe Holistic, recorta cada seña,
la lleva a 30 fotogramas y guarda:
    data/X.npy  (N, 30, 258)  landmarks de cada seña
    data/y.npy  (N,)          nombre de la seña

Datasets (ver README.md para los enlaces):
    PERUSIL  -> LSP PUCP-DGI156          (un video por seña, el nombre del archivo es la seña)
    APRENDO  -> LSP Aprendo en Casa 2020 (un video por seña, el nombre del archivo es la seña)
    PUCP     -> LSP PUCP 305             (una carpeta por seña con videos aislados y oraciones con .eaf)

Ejecutar desde la carpeta modelo/ (antes, cambiar las tres rutas de abajo):
    python src/procesar_todo.py
"""
import os, glob, unicodedata
from collections import Counter
import cv2
import numpy as np
import mediapipe as mp
from keypoints import extraer_keypoints
from eaf import leer_anotaciones

# ===== CONFIGURACIÓN =====
SALIDA_DIR = "data"
N_FRAMES = 30               # todas las señas se llevan a 30 fotogramas
PERUSIL = r"C:\Users\Joel\Downloads\XXXX_LSP_peru1235_Videos\Videos\SEGMENTED_SIGN_ADJUSTED"
APRENDO = r"C:\Users\Joel\Downloads\2020-2021_LSP_peru1235_videos\Videos\SEGMENTED_SIGN"
PUCP    = r"C:\Users\Joel\Downloads\2015-2023_LSP_peru1235_PUCP305_glosas (1)\5. Segundo avance (corregido)"

# Señas que se extraen (18). Para entrenar se eligen 10 de ellas en comun.SEÑAS_USAR.
OBJETIVO = {"YO","PENSAR","MUJER","HOMBRE","CASA","VER","ESPERAR","CAMINAR","QUE","MAMA",
            "HERMANO","COMER","IR","DECIR","PREGUNTAR","DONDE","NO","BIEN"}

mp_holistic = mp.solutions.holistic


def norm(s):
    """Unifica el nombre de una seña: quita tildes (NFKD + ASCII), pasa a mayúsculas y quita espacios.
    Ej.: "mamá " -> "MAMA"."""
    return unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().upper().strip()


def ajustar_longitud(sec, n=N_FRAMES):
    """Lleva una lista de fotogramas a exactamente n fotogramas tomando índices equiespaciados.
    Devuelve un array (n, 258); si la lista está vacía, devuelve ceros."""
    sec = np.array(sec)
    if len(sec) == 0: return np.zeros((n, 258))
    idx = np.linspace(0, len(sec) - 1, n).astype(int)   # ej. 90 fotogramas -> 0, 3, 6, ..., 89
    return sec[idx]


def keypoints_video(ruta, holistic):
    """Procesa un video completo con MediaPipe Holistic.
    Devuelve una lista con un elemento por fotograma: (tiempo_ms, vector_258, hay_manos)."""
    cap = cv2.VideoCapture(ruta); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break                                        # fin del video
        t = i / fps * 1000                                      # tiempo del fotograma en ms
        res = holistic.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))   # OpenCV lee en BGR, MediaPipe usa RGB
        manos = bool(res.left_hand_landmarks or res.right_hand_landmarks)
        frames.append((t, extraer_keypoints(res), manos)); i += 1
    cap.release()
    return frames


def recorte(frames, ini=None, fin=None):
    """Recorta una seña de la lista de fotogramas y la lleva a 30 fotogramas.

    ini, fin: intervalo en ms (de la anotación .eaf). Si son None se usa todo el video.
    Dentro del intervalo se usan solo los fotogramas con manos visibles; si hay menos de 5,
    se usan todos para no perder la muestra.
    """
    sub = [(t, kp, m) for (t, kp, m) in frames if (ini is None or ini <= t <= fin)]   # fotogramas del intervalo
    cm = [kp for (t, kp, m) in sub if m]                                                # solo los que tienen manos
    todos = [kp for (t, kp, m) in sub]
    return ajustar_longitud(cm if len(cm) >= 5 else todos)


def main():
    X, y = [], []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # ---- 1. PUCP-DGI156 y Aprendo en Casa: la etiqueta sale del nombre del archivo (ej. CASA_3.mp4 -> CASA)
        for base in (PERUSIL, APRENDO):
            vids = glob.glob(os.path.join(base, "**", "*.mp4"), recursive=True)
            print(f"Procesando {os.path.basename(base)} ({len(vids)} videos)...")
            for v in vids:
                if norm(os.path.basename(v)[:-4].rsplit("_", 1)[0]) not in OBJETIVO: continue
                X.append(recorte(keypoints_video(v, holistic)))
                y.append(norm(os.path.basename(v)[:-4].rsplit("_", 1)[0]))
                if len(y) % 50 == 0: print(f"  acumulado {len(y)}")
        # ---- 2. PUCP 305: una carpeta por seña
        carps = sorted(d for d in os.listdir(PUCP) if os.path.isdir(os.path.join(PUCP, d)))
        print(f"Procesando PUCP ({len(carps)} carpetas)...")
        for carp in carps:
            d = os.path.join(PUCP, carp)
            vids = glob.glob(os.path.join(d, "*.mp4"))
            aislados = [v for v in vids if "ORACION" not in os.path.basename(v).upper()]
            oraciones = [v for v in vids if "ORACION" in os.path.basename(v).upper()]
            # 2a. Videos aislados: la etiqueta es el nombre de la carpeta
            if norm(carp) in OBJETIVO:
                for v in aislados:
                    X.append(recorte(keypoints_video(v, holistic))); y.append(norm(carp))
            # 2b. Oraciones: se recortan las señas objetivo usando los tiempos del .eaf
            for v in oraciones:
                eaf = v.replace(".mp4", ".eaf")
                if not os.path.exists(eaf): continue
                objs = [(norm(et.rsplit("_", 1)[0]), i, f) for et, i, f in leer_anotaciones(eaf)]
                objs = [(b, i, f) for (b, i, f) in objs if b in OBJETIVO]
                if not objs: continue                           # la oración no tiene señas objetivo
                frames = keypoints_video(v, holistic)           # el video se procesa una sola vez
                for b, i, f in objs:
                    X.append(recorte(frames, i, f)); y.append(b)
    # ---- 3. Guardar
    X, y = np.array(X), np.array(y)
    os.makedirs(SALIDA_DIR, exist_ok=True)
    np.save(os.path.join(SALIDA_DIR, "X.npy"), X)
    np.save(os.path.join(SALIDA_DIR, "y.npy"), y)
    print(f"\nListo. X={X.shape}")
    for s, n in sorted(Counter(y).items(), key=lambda x: -x[1]):    # muestras por seña, de mayor a menor
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
