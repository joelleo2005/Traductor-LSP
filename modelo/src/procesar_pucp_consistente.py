import os, glob
from collections import Counter
import cv2
import numpy as np
import mediapipe as mp
from keypoints import extraer_keypoints
from eaf import leer_anotaciones

PUCP_DIR = r"C:\Users\Joel\Downloads\2015-2023_LSP_peru1235_PUCP305_glosas (1)\5. Segundo avance (corregido)"
SALIDA_DIR = "data"
N_FRAMES = 30
OBJETIVO = {"YO","TÚ","ELLA","MUJER","HOMBRE","MAMÁ","PENSAR","VER","CAMINAR","IR",
            "ENTRAR","ESPERAR","CASA","AHÍ","DOS","UNO","QUÉ","NO","SÍ","LUEGO"}

mp_holistic = mp.solutions.holistic


def ajustar_longitud(sec, n=N_FRAMES):
    sec = np.array(sec)
    if len(sec) == 0:
        return np.zeros((n, 258))
    idx = np.linspace(0, len(sec) - 1, n).astype(int)
    return sec[idx]


def keypoints_por_frame(ruta, holistic):
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / fps * 1000
        image = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        res = holistic.process(image)
        manos = bool(res.left_hand_landmarks or res.right_hand_landmarks)
        frames.append((t, extraer_keypoints(res), manos))
        i += 1
    cap.release()
    return frames


def recortar_manos(frames, ini=None, fin=None):
    sub = [(t, kp, m) for (t, kp, m) in frames if (ini is None or ini <= t <= fin)]
    con_manos = [kp for (t, kp, m) in sub if m]
    todos = [kp for (t, kp, m) in sub]
    seq = con_manos if len(con_manos) >= 5 else todos
    return ajustar_longitud(seq)


def main():
    carpetas = sorted(d for d in os.listdir(PUCP_DIR) if os.path.isdir(os.path.join(PUCP_DIR, d)))
    X, y = [], []
    procesados = 0
    total_carpetas = len(carpetas)
    for n, sena in enumerate(carpetas, 1):
        print(f"[{n}/{total_carpetas}] carpeta '{sena}'  (muestras hasta ahora: {len(X)})")
        ruta = os.path.join(PUCP_DIR, sena)
        videos = glob.glob(os.path.join(ruta, "*.mp4"))
        aislados  = [v for v in videos if "ORACION" not in os.path.basename(v).upper()]
        oraciones = [v for v in videos if "ORACION" in os.path.basename(v).upper()]

        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            if sena in OBJETIVO:
                for v in aislados:
                    frames = keypoints_por_frame(v, holistic)
                    X.append(recortar_manos(frames)); y.append(sena)
            for v in oraciones:
                eaf = v.replace(".mp4", ".eaf")
                if not os.path.exists(eaf):
                    continue
                objetivo_en = [(et.rsplit("_", 1)[0], i, f)
                               for et, i, f in leer_anotaciones(eaf)
                               if et.rsplit("_", 1)[0] in OBJETIVO]
                if not objetivo_en:
                    continue
                frames = keypoints_por_frame(v, holistic)
                for base, ini, fin in objetivo_en:
                    X.append(recortar_manos(frames, ini, fin)); y.append(base)

    X, y = np.array(X), np.array(y)
    np.save(os.path.join(SALIDA_DIR, "X_pucp.npy"), X)
    np.save(os.path.join(SALIDA_DIR, "y_pucp.npy"), y)
    print(f"\nListo. X_pucp={X.shape}")
    print("Por seña:", dict(Counter(y)))


if __name__ == "__main__":
    main()