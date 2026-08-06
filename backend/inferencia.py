import cv2
import numpy as np
import mediapipe as mp
from ai_edge_litert.interpreter import Interpreter

from keypoints import extraer_keypoints
from normalizar import normalizar

N_FRAMES = 30
mp_holistic = mp.solutions.holistic

with open("clases.txt", encoding="utf-8") as f:
    CLASES = [linea.strip() for linea in f if linea.strip()]

interpreter = Interpreter(model_path="modelo_lsp.tflite")
interpreter.allocate_tensors()
entrada = interpreter.get_input_details()[0]
salida  = interpreter.get_output_details()[0]


def ajustar_longitud(secuencia, n=N_FRAMES):
    secuencia = np.array(secuencia)
    if len(secuencia) == 0:
        return np.zeros((n, 258))
    idx = np.linspace(0, len(secuencia) - 1, n).astype(int)
    return secuencia[idx]


def extraer_secuencia(video_path):
    cap = cv2.VideoCapture(video_path)
    todos, con_manos, total = [], [], 0
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            kp = extraer_keypoints(results)
            todos.append(kp)
            if results.left_hand_landmarks or results.right_hand_landmarks:
                con_manos.append(kp)   # frame donde SÍ hay señas de manos
    cap.release()
    # Recortar a los frames con manos (aproxima el recorte del .eaf); si casi no hay, usar todos
    frames = con_manos if len(con_manos) >= 5 else todos
    return ajustar_longitud(frames), len(con_manos), total


def predecir(video_path):
    secuencia, manos, total = extraer_secuencia(video_path)
    X = np.expand_dims(secuencia, axis=0).astype("float32")
    X = normalizar(X).astype("float32")

    interpreter.set_tensor(entrada["index"], X)
    interpreter.invoke()
    probs = interpreter.get_tensor(salida["index"])[0]

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(CLASES[i], round(float(probs[i]) * 100, 1)) for i in top3_idx]
    return {
        "sena": CLASES[int(np.argmax(probs))],
        "confianza": float(np.max(probs)),
        "manos_detectadas": f"{manos}/{total} frames",
        "top3": top3,
    }