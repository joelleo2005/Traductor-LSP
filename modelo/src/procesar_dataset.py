import os, glob
import cv2
import numpy as np
import mediapipe as mp

from keypoints import extraer_keypoints
from eaf import leer_anotaciones

# ===== CONFIGURACIÓN =====
DATASET_DIR = r"C:\Users\Joel\Downloads\2015-2023_LSP_peru1235_PUCP305_glosas (1)\5. Segundo avance (corregido)"
SALIDA_DIR  = "data"      # se guarda en modelo/data (Git lo ignora)
N_FRAMES    = 30          # longitud fija de cada secuencia
MAX_CARPETAS = None        # <-- PRUEBA con pocas; luego pon None para las 142

mp_holistic = mp.solutions.holistic


def ajustar_longitud(secuencia, n=N_FRAMES):
    secuencia = np.array(secuencia)
    if len(secuencia) == 0:
        return np.zeros((n, 258))
    idx = np.linspace(0, len(secuencia) - 1, n).astype(int)  # n frames equiespaciados
    return secuencia[idx]


def procesar_video(ruta_mp4, ruta_eaf, holistic):
    # Ventana de tiempo de la seña (del .eaf); si no hay, usa todo el video
    ini = fin = None
    if os.path.exists(ruta_eaf):
        anot = leer_anotaciones(ruta_eaf)
        if anot:
            _, ini, fin = anot[0]

    cap = cv2.VideoCapture(ruta_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    secuencia, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps * 1000  # ms del frame actual
        if ini is None or ini <= t <= fin:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            secuencia.append(extraer_keypoints(results))
        i += 1
    cap.release()
    return ajustar_longitud(secuencia)


def main():
    carpetas = sorted(d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d)))
    if MAX_CARPETAS:
        carpetas = carpetas[:MAX_CARPETAS]

    X, y = [], []
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        for seña in carpetas:
            ruta = os.path.join(DATASET_DIR, seña)
            videos = [v for v in glob.glob(os.path.join(ruta, "*.mp4"))
                      if "ORACION" not in os.path.basename(v).upper()]
            print(f"Procesando '{seña}' ({len(videos)} videos)...")
            for ruta_mp4 in videos:
                ruta_eaf = ruta_mp4.replace(".mp4", ".eaf")
                X.append(procesar_video(ruta_mp4, ruta_eaf, holistic))
                y.append(seña)   # etiqueta = nombre de la carpeta

    X, y = np.array(X), np.array(y)
    os.makedirs(SALIDA_DIR, exist_ok=True)
    np.save(os.path.join(SALIDA_DIR, "X.npy"), X)
    np.save(os.path.join(SALIDA_DIR, "y.npy"), y)
    print(f"\nListo. X={X.shape}  y={y.shape}")
    print("Señas:", sorted(set(y)))


if __name__ == "__main__":
    main()