import os, glob
from collections import Counter
import cv2
import numpy as np
import mediapipe as mp
from keypoints import extraer_keypoints

# ===== CONFIGURACIÓN =====
PERUSIL_DIR = r"C:\Users\Joel\Downloads\XXXX_LSP_peru1235_Videos\Videos\SEGMENTED_SIGN_ADJUSTED"
SALIDA_DIR  = "data"
N_FRAMES    = 30

# Las 20 señas objetivo (nombres exactos de PeruSIL)
OBJETIVO = {"YO","TÚ","ELLA","MUJER","HOMBRE","MAMÁ","PENSAR","VER","CAMINAR","IR",
            "ENTRAR","ESPERAR","CASA","AHÍ","DOS","UNO","QUÉ","NO","SÍ","LUEGO"}

mp_holistic = mp.solutions.holistic


def etiqueta_de(ruta):
    base = os.path.basename(ruta)[:-4]      # sin .mp4
    et = base.rsplit("_", 1)[0]             # quita el _NNNN final
    return et.rstrip("_").strip()           # limpia guiones bajos sobrantes


def ajustar_longitud(secuencia, n=N_FRAMES):
    secuencia = np.array(secuencia)
    if len(secuencia) == 0:
        return np.zeros((n, 258))
    idx = np.linspace(0, len(secuencia) - 1, n).astype(int)
    return secuencia[idx]


def extraer_secuencia(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    todos, con_manos = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        kp = extraer_keypoints(results)
        todos.append(kp)
        if results.left_hand_landmarks or results.right_hand_landmarks:
            con_manos.append(kp)
    cap.release()
    frames = con_manos if len(con_manos) >= 5 else todos
    return ajustar_longitud(frames)


def main():
    videos = glob.glob(os.path.join(PERUSIL_DIR, "**", "*.mp4"), recursive=True)
    X, y, procesados = [], [], 0
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        for v in videos:
            if etiqueta_de(v) not in OBJETIVO:
                continue
            X.append(extraer_secuencia(v, holistic))
            y.append(etiqueta_de(v))
            procesados += 1
            if procesados % 25 == 0:
                print(f"  {procesados} videos procesados...")
    X, y = np.array(X), np.array(y)
    os.makedirs(SALIDA_DIR, exist_ok=True)
    np.save(os.path.join(SALIDA_DIR, "X_perusil.npy"), X)
    np.save(os.path.join(SALIDA_DIR, "y_perusil.npy"), y)
    print(f"\nListo. X_perusil={X.shape}  y={y.shape}")
    print("Por seña:", dict(Counter(y)))


if __name__ == "__main__":
    main()