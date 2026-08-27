import os, glob
from collections import Counter
import cv2
import numpy as np
import mediapipe as mp
from keypoints import extraer_keypoints

APRENDO_DIR = r"C:\Users\Joel\Downloads\2020-2021_LSP_peru1235_videos\Videos\SEGMENTED_SIGN"
SALIDA_DIR = "data"
N_FRAMES = 30

# Mapear nombres (minúsculas) a tus etiquetas (MAYÚSCULAS)
MAPA = {
    "yo": "YO", "ver": "VER", "pensar": "PENSAR", "casa": "CASA",
    "que": "QUÉ", "qué": "QUÉ", "mama": "MAMÁ", "mamá": "MAMÁ",
    "mujer": "MUJER", "hombre": "HOMBRE", "esperar": "ESPERAR", "caminar": "CAMINAR",
}

mp_holistic = mp.solutions.holistic

def etiqueta_de(ruta):
    return os.path.basename(ruta)[:-4].rsplit("_", 1)[0].strip().lower()

def ajustar_longitud(sec, n=N_FRAMES):
    sec = np.array(sec)
    if len(sec) == 0: return np.zeros((n, 258))
    idx = np.linspace(0, len(sec) - 1, n).astype(int)
    return sec[idx]

def extraer_secuencia(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    todos, con_manos = [], []
    while True:
        ok, frame = cap.read()
        if not ok: break
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
    videos = glob.glob(os.path.join(APRENDO_DIR, "**", "*.mp4"), recursive=True)
    X, y, n = [], [], 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for v in videos:
            et = etiqueta_de(v)
            if et not in MAPA: continue
            X.append(extraer_secuencia(v, holistic))
            y.append(MAPA[et])
            n += 1
            if n % 20 == 0: print(f"  {n} videos procesados...")
    X, y = np.array(X), np.array(y)
    np.save(os.path.join(SALIDA_DIR, "X_aprendo.npy"), X)
    np.save(os.path.join(SALIDA_DIR, "y_aprendo.npy"), y)
    print(f"\nListo. X_aprendo={X.shape}")
    print("Por seña:", dict(Counter(y)))

if __name__ == "__main__":
    main()