import os, glob
import cv2
import numpy as np
import mediapipe as mp

from keypoints import extraer_keypoints
from eaf import leer_anotaciones

# ===== CONFIGURACIÓN =====
DATASET_DIR = r"C:\Users\Joel\Downloads\2015-2023_LSP_peru1235_PUCP305_glosas (1)\5. Segundo avance (corregido)"
SALIDA_DIR  = "data"
N_FRAMES    = 30
MAX_CARPETAS = None

mp_holistic = mp.solutions.holistic


def ajustar_longitud(secuencia, n=N_FRAMES):
    secuencia = np.array(secuencia)
    if len(secuencia) == 0:
        return np.zeros((n, 258))
    idx = np.linspace(0, len(secuencia) - 1, n).astype(int)
    return secuencia[idx]


def keypoints_por_frame(ruta_mp4, holistic):
    """Procesa TODO el video una vez -> [(t_ms, keypoints), ...]."""
    cap = cv2.VideoCapture(ruta_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps * 1000
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        frames.append((t, extraer_keypoints(results)))
        i += 1
    cap.release()
    return frames


def recortar(frames, ini, fin):
    """Secuencia (30,258) del tramo [ini, fin] ms."""
    sub = [kp for (t, kp) in frames if ini <= t <= fin]
    return ajustar_longitud(sub)


def main():
    carpetas = sorted(d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d)))
    if MAX_CARPETAS:
        carpetas = carpetas[:MAX_CARPETAS]
    clases = set(carpetas)

    X, y = [], []
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        for seña in carpetas:
            ruta = os.path.join(DATASET_DIR, seña)
            videos = glob.glob(os.path.join(ruta, "*.mp4"))
            aislados  = [v for v in videos if "ORACION" not in os.path.basename(v).upper()]
            oraciones = [v for v in videos if "ORACION" in os.path.basename(v).upper()]
            print(f"'{seña}': {len(aislados)} aislados + {len(oraciones)} oraciones")

            # --- Videos de seña aislada ---
            for v in aislados:
                frames = keypoints_por_frame(v, holistic)
                anot = leer_anotaciones(v.replace(".mp4", ".eaf"))
                if anot:
                    _, ini, fin = anot[0]
                    sec = recortar(frames, ini, fin)
                else:
                    sec = ajustar_longitud([kp for _, kp in frames])
                X.append(sec); y.append(seña)

            # --- Videos de oración: extraemos cada seña-clase que contengan ---
            for v in oraciones:
                eaf = v.replace(".mp4", ".eaf")
                if not os.path.exists(eaf):
                    continue
                frames = keypoints_por_frame(v, holistic)
                for etiqueta, ini, fin in leer_anotaciones(eaf):
                    base = etiqueta.rsplit("_", 1)[0]     # "AMIGO_1" -> "AMIGO"
                    if base in clases:                    # solo si es una de nuestras señas
                        sec = recortar(frames, ini, fin)
                        X.append(sec); y.append(base)

    X, y = np.array(X), np.array(y)
    os.makedirs(SALIDA_DIR, exist_ok=True)
    np.save(os.path.join(SALIDA_DIR, "X.npy"), X)
    np.save(os.path.join(SALIDA_DIR, "y.npy"), y)
    print(f"\nListo. X={X.shape}  y={y.shape}")
    print("Clases:", len(set(y)))


if __name__ == "__main__":
    main()