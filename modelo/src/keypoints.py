"""
Conversión de la salida de MediaPipe Holistic a un vector numérico por fotograma.

Vector de 258 valores:
    [0   : 132)  pose          33 puntos x (x, y, z, visibilidad)
    [132 : 195)  mano izquierda 21 puntos x (x, y, z)
    [195 : 258)  mano derecha   21 puntos x (x, y, z)
Si MediaPipe no detecta una parte del cuerpo en el fotograma, esa parte se rellena con ceros.
"""
import numpy as np


def extraer_keypoints(results):
    """Recibe el resultado de holistic.process(imagen) y devuelve un np.array de 258 valores."""
    # Pose: 33 puntos con x, y, z y visibilidad (probabilidad de que el punto sea visible)
    pose = np.array([[p.x, p.y, p.z, p.visibility]
                     for p in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)
    # Manos: 21 puntos con x, y, z (MediaPipe no da visibilidad para las manos)
    mano_izq = np.array([[p.x, p.y, p.z]
                         for p in results.left_hand_landmarks.landmark]).flatten() \
               if results.left_hand_landmarks else np.zeros(21 * 3)
    mano_der = np.array([[p.x, p.y, p.z]
                         for p in results.right_hand_landmarks.landmark]).flatten() \
               if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, mano_izq, mano_der])   # 132 + 63 + 63 = 258
