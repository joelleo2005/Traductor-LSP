import os, glob, unicodedata
from collections import Counter
import cv2
import numpy as np
import mediapipe as mp
from keypoints import extraer_keypoints
from eaf import leer_anotaciones

SALIDA_DIR = "data"
N_FRAMES = 30
PERUSIL = r"C:\Users\Joel\Downloads\XXXX_LSP_peru1235_Videos\Videos\SEGMENTED_SIGN_ADJUSTED"
APRENDO = r"C:\Users\Joel\Downloads\2020-2021_LSP_peru1235_videos\Videos\SEGMENTED_SIGN"
PUCP    = r"C:\Users\Joel\Downloads\2015-2023_LSP_peru1235_PUCP305_glosas (1)\5. Segundo avance (corregido)"

OBJETIVO = {"YO","PENSAR","MUJER","HOMBRE","CASA","VER","ESPERAR","CAMINAR","QUE","MAMA",
            "HERMANO","COMER","IR","DECIR","PREGUNTAR","DONDE","NO","BIEN"}

mp_holistic = mp.solutions.holistic

def norm(s):
    return unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().upper().strip()

def ajustar_longitud(sec, n=N_FRAMES):
    sec = np.array(sec)
    if len(sec) == 0: return np.zeros((n, 258))
    idx = np.linspace(0, len(sec) - 1, n).astype(int)
    return sec[idx]

def keypoints_video(ruta, holistic):
    cap = cv2.VideoCapture(ruta); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        t = i / fps * 1000
        res = holistic.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        manos = bool(res.left_hand_landmarks or res.right_hand_landmarks)
        frames.append((t, extraer_keypoints(res), manos)); i += 1
    cap.release()
    return frames

def recorte(frames, ini=None, fin=None):
    sub = [(t, kp, m) for (t, kp, m) in frames if (ini is None or ini <= t <= fin)]
    cm = [kp for (t, kp, m) in sub if m]
    todos = [kp for (t, kp, m) in sub]
    return ajustar_longitud(cm if len(cm) >= 5 else todos)

def main():
    X, y = [], []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for base in (PERUSIL, APRENDO):
            vids = glob.glob(os.path.join(base, "**", "*.mp4"), recursive=True)
            print(f"Procesando {os.path.basename(base)} ({len(vids)} videos)...")
            for v in vids:
                if norm(os.path.basename(v)[:-4].rsplit("_", 1)[0]) not in OBJETIVO: continue
                X.append(recorte(keypoints_video(v, holistic)))
                y.append(norm(os.path.basename(v)[:-4].rsplit("_", 1)[0]))
                if len(y) % 50 == 0: print(f"  acumulado {len(y)}")
        carps = sorted(d for d in os.listdir(PUCP) if os.path.isdir(os.path.join(PUCP, d)))
        print(f"Procesando PUCP ({len(carps)} carpetas)...")
        for carp in carps:
            d = os.path.join(PUCP, carp)
            vids = glob.glob(os.path.join(d, "*.mp4"))
            aislados = [v for v in vids if "ORACION" not in os.path.basename(v).upper()]
            oraciones = [v for v in vids if "ORACION" in os.path.basename(v).upper()]
            if norm(carp) in OBJETIVO:
                for v in aislados:
                    X.append(recorte(keypoints_video(v, holistic))); y.append(norm(carp))
            for v in oraciones:
                eaf = v.replace(".mp4", ".eaf")
                if not os.path.exists(eaf): continue
                objs = [(norm(et.rsplit("_", 1)[0]), i, f) for et, i, f in leer_anotaciones(eaf)]
                objs = [(b, i, f) for (b, i, f) in objs if b in OBJETIVO]
                if not objs: continue
                frames = keypoints_video(v, holistic)
                for b, i, f in objs:
                    X.append(recorte(frames, i, f)); y.append(b)
    X, y = np.array(X), np.array(y)
    os.makedirs(SALIDA_DIR, exist_ok=True)
    np.save(os.path.join(SALIDA_DIR, "X.npy"), X)
    np.save(os.path.join(SALIDA_DIR, "y.npy"), y)
    print(f"\nListo. X={X.shape}")
    for s, n in sorted(Counter(y).items(), key=lambda x: -x[1]):
        print(f"  {s}: {n}")

if __name__ == "__main__":
    main()